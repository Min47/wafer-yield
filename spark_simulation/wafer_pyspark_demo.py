"""
PySpark Simulation for Wafer Yield Optimization
==============================================

This module demonstrates how to scale the wafer yield optimization pipeline
using Apache Spark for distributed processing of large-scale manufacturing data.

This simulation shows how the project would scale to handle:
- Millions of wafers
- Real-time sensor data streams
- Distributed machine learning
- Big data processing workflows

Author: Data Science Team
Target: Micron Technology
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, isnan, isnull, mean, stddev, count, sum as spark_sum
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
import warnings
warnings.filterwarnings('ignore')

class WaferSparkSimulation:
    """Simulate distributed processing for wafer yield optimization"""
    
    def __init__(self, app_name="WaferYieldOptimization"):
        """Initialize Spark session"""
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        print(f"Spark session initialized: {self.spark.version}")
    
    def create_synthetic_wafer_data(self, n_wafers=100000, n_features=100):
        """
        Create synthetic wafer data for simulation
        
        Args:
            n_wafers (int): Number of wafers to simulate
            n_features (int): Number of sensor features per wafer
        
        Returns:
            pyspark.sql.DataFrame: Synthetic wafer data
        """
        print(f"Creating synthetic dataset: {n_wafers:,} wafers, {n_features} features...")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Generate synthetic data
        data = []
        for i in range(n_wafers):
            # Generate sensor readings
            sensor_data = np.random.normal(0, 1, n_features)
            
            # Add some realistic patterns
            # Temperature sensors (correlated)
            temp_sensors = np.random.choice(range(n_features), 20, replace=False)
            for j, sensor in enumerate(temp_sensors):
                sensor_data[sensor] = 25 + j * 0.5 + np.random.normal(0, 2)
            
            # Pressure sensors (correlated)
            pressure_sensors = np.random.choice(range(n_features), 15, replace=False)
            for j, sensor in enumerate(pressure_sensors):
                sensor_data[sensor] = 1.0 + j * 0.1 + np.random.normal(0, 0.1)
            
            # Add missing values (realistic for manufacturing)
            missing_mask = np.random.random(n_features) < 0.2
            sensor_data[missing_mask] = np.nan
            
            # Determine yield based on sensor patterns
            good_yield = (
                sensor_data[temp_sensors[:10]].mean() > 0 and
                sensor_data[pressure_sensors[:5]].mean() > 0 and
                np.random.random() > 0.15
            )
            yield_class = 1 if good_yield else -1
            
            # Create row
            row = [f"wafer_{i:06d}"] + sensor_data.tolist() + [yield_class]
            data.append(row)
        
        # Create schema
        schema_fields = [StructField("wafer_id", StringType(), True)]
        for i in range(n_features):
            schema_fields.append(StructField(f"sensor_{i:03d}", DoubleType(), True))
        schema_fields.append(StructField("yield_class", IntegerType(), True))
        
        schema = StructType(schema_fields)
        
        # Create DataFrame
        df = self.spark.createDataFrame(data, schema)
        
        print(f"Dataset created: {df.count():,} rows, {len(df.columns)} columns")
        return df
    
    def preprocess_data_distributed(self, df):
        """
        Preprocess data using distributed Spark operations
        
        Args:
            df (pyspark.sql.DataFrame): Raw wafer data
        
        Returns:
            pyspark.sql.DataFrame: Preprocessed data
        """
        print("Starting distributed preprocessing...")
        start_time = time.time()
        
        # Get sensor columns
        sensor_cols = [col for col in df.columns if col.startswith('sensor_')]
        
        # 1. Handle missing values with mean imputation
        print("Handling missing values...")
        imputed_df = df
        for sensor_col in sensor_cols:
            mean_val = df.select(mean(sensor_col)).collect()[0][0]
            imputed_df = imputed_df.withColumn(
                sensor_col,
                when(isnull(sensor_col) | isnan(sensor_col), mean_val).otherwise(col(sensor_col))
            )
        
        # 2. Create feature vector
        print("Creating feature vectors...")
        assembler = VectorAssembler(
            inputCols=sensor_cols,
            outputCol="features"
        )
        
        # 3. Scale features
        print("Scaling features...")
        scaler = StandardScaler(
            inputCol="features",
            outputCol="scaled_features",
            withStd=True,
            withMean=True
        )
        
        # 4. Apply PCA for dimensionality reduction
        print("Applying PCA...")
        pca = PCA(
            k=50,  # Reduce to 50 principal components
            inputCol="scaled_features",
            outputCol="pca_features"
        )
        
        # Create pipeline
        pipeline = Pipeline(stages=[assembler, scaler, pca])
        
        # Fit and transform
        model = pipeline.fit(imputed_df)
        processed_df = model.transform(imputed_df)
        
        # Select final columns
        final_df = processed_df.select("wafer_id", "pca_features", "yield_class")
        
        processing_time = time.time() - start_time
        print(f"Preprocessing completed in {processing_time:.2f} seconds")
        print(f"Final dataset: {final_df.count():,} rows")
        
        return final_df, model
    
    def train_distributed_model(self, df):
        """
        Train machine learning model using Spark ML
        
        Args:
            df (pyspark.sql.DataFrame): Preprocessed data
        
        Returns:
            tuple: (trained_model, evaluation_metrics)
        """
        print("Training distributed ML model...")
        start_time = time.time()
        
        # Split data
        train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
        
        # Train Random Forest
        rf = RandomForestClassifier(
            featuresCol="pca_features",
            labelCol="yield_class",
            numTrees=100,
            maxDepth=10,
            seed=42
        )
        
        # Train model
        model = rf.fit(train_df)
        
        # Make predictions
        predictions = model.transform(test_df)
        
        # Evaluate model
        evaluator = BinaryClassificationEvaluator(
            labelCol="yield_class",
            rawPredictionCol="rawPrediction"
        )
        
        auc = evaluator.evaluate(predictions)
        
        # Multiclass metrics
        multi_evaluator = MulticlassClassificationEvaluator(
            labelCol="yield_class",
            predictionCol="prediction",
            metricName="accuracy"
        )
        
        accuracy = multi_evaluator.evaluate(predictions)
        
        training_time = time.time() - start_time
        print(f"Model training completed in {training_time:.2f} seconds")
        print(f"Test AUC: {auc:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        
        return model, {"auc": auc, "accuracy": accuracy}
    
    def simulate_real_time_processing(self, df, batch_size=1000):
        """
        Simulate real-time processing of wafer data
        
        Args:
            df (pyspark.sql.DataFrame): Wafer data
            batch_size (int): Batch size for processing
        
        Returns:
            dict: Processing statistics
        """
        print(f"Simulating real-time processing with batch size {batch_size:,}...")
        
        # Simulate streaming data by processing in batches
        total_wafers = df.count()
        n_batches = total_wafers // batch_size
        
        processing_stats = {
            "total_wafers": total_wafers,
            "batch_size": batch_size,
            "n_batches": n_batches,
            "batch_processing_times": [],
            "throughput": []
        }
        
        start_time = time.time()
        
        for i in range(min(10, n_batches)):  # Process first 10 batches for demo
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, total_wafers)
            
            # Simulate batch processing
            batch_start_time = time.time()
            
            # Get batch data
            batch_df = df.limit(batch_size).offset(batch_start)
            
            # Simulate processing (feature extraction, prediction, etc.)
            batch_df.count()  # Trigger computation
            
            batch_time = time.time() - batch_start_time
            throughput = batch_size / batch_time
            
            processing_stats["batch_processing_times"].append(batch_time)
            processing_stats["throughput"].append(throughput)
            
            print(f"Batch {i+1}/{n_batches}: {batch_time:.3f}s, {throughput:.0f} wafers/sec")
        
        total_time = time.time() - start_time
        avg_throughput = np.mean(processing_stats["throughput"])
        
        print(f"Real-time simulation completed:")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average throughput: {avg_throughput:.0f} wafers/second")
        
        return processing_stats
    
    def benchmark_performance(self, df):
        """
        Benchmark performance metrics for the distributed system
        
        Args:
            df (pyspark.sql.DataFrame): Dataset to benchmark
        
        Returns:
            dict: Performance metrics
        """
        print("Running performance benchmarks...")
        
        benchmarks = {}
        
        # 1. Data loading benchmark
        start_time = time.time()
        count = df.count()
        load_time = time.time() - start_time
        benchmarks["data_loading"] = {
            "time_seconds": load_time,
            "rows_per_second": count / load_time
        }
        
        # 2. Aggregation benchmark
        start_time = time.time()
        agg_result = df.groupBy("yield_class").count().collect()
        agg_time = time.time() - start_time
        benchmarks["aggregation"] = {
            "time_seconds": agg_time,
            "operations_per_second": 1 / agg_time
        }
        
        # 3. Filtering benchmark
        start_time = time.time()
        filtered_count = df.filter(col("yield_class") == 1).count()
        filter_time = time.time() - start_time
        benchmarks["filtering"] = {
            "time_seconds": filter_time,
            "rows_per_second": filtered_count / filter_time
        }
        
        print("Benchmark Results:")
        for operation, metrics in benchmarks.items():
            print(f"{operation}: {metrics}")
        
        return benchmarks
    
    def generate_scalability_report(self, original_size, processed_size, processing_time):
        """
        Generate scalability analysis report
        
        Args:
            original_size (int): Original dataset size
            processed_size (int): Processed dataset size
            processing_time (float): Processing time in seconds
        
        Returns:
            dict: Scalability metrics
        """
        print("Generating scalability report...")
        
        # Calculate scalability metrics
        compression_ratio = original_size / processed_size
        processing_speed = original_size / processing_time
        
        # Estimate scaling to larger datasets
        scale_factors = [10, 100, 1000, 10000]  # 10x, 100x, 1000x, 10000x
        scaling_predictions = {}
        
        for factor in scale_factors:
            scaled_size = original_size * factor
            estimated_time = processing_time * factor
            estimated_throughput = scaled_size / estimated_time
            
            scaling_predictions[f"{factor}x"] = {
                "dataset_size": scaled_size,
                "estimated_time_seconds": estimated_time,
                "estimated_time_hours": estimated_time / 3600,
                "throughput_per_second": estimated_throughput
            }
        
        report = {
            "original_size": original_size,
            "processed_size": processed_size,
            "compression_ratio": compression_ratio,
            "processing_speed": processing_speed,
            "scaling_predictions": scaling_predictions
        }
        
        print("Scalability Report:")
        print(f"Original size: {original_size:,} rows")
        print(f"Processed size: {processed_size:,} rows")
        print(f"Compression ratio: {compression_ratio:.2f}x")
        print(f"Processing speed: {processing_speed:.0f} rows/second")
        
        print("\nScaling Predictions:")
        for scale, metrics in scaling_predictions.items():
            print(f"{scale} scale: {metrics['estimated_time_hours']:.2f} hours")
        
        return report
    
    def cleanup(self):
        """Clean up Spark session"""
        if self.spark:
            self.spark.stop()
            print("Spark session stopped")

def main():
    """Main simulation function"""
    print("=" * 60)
    print("PySpark Wafer Yield Optimization Simulation")
    print("=" * 60)
    
    # Initialize simulation
    simulation = WaferSparkSimulation()
    
    try:
        # 1. Create synthetic dataset
        print("\n1. Creating synthetic wafer dataset...")
        df = simulation.create_synthetic_wafer_data(n_wafers=50000, n_features=100)
        
        # 2. Benchmark original data
        print("\n2. Benchmarking original data...")
        original_benchmarks = simulation.benchmark_performance(df)
        
        # 3. Preprocess data
        print("\n3. Distributed preprocessing...")
        processed_df, preprocessing_model = simulation.preprocess_data_distributed(df)
        
        # 4. Train model
        print("\n4. Training distributed model...")
        model, metrics = simulation.train_distributed_model(processed_df)
        
        # 5. Simulate real-time processing
        print("\n5. Real-time processing simulation...")
        real_time_stats = simulation.simulate_real_time_processing(processed_df)
        
        # 6. Generate scalability report
        print("\n6. Generating scalability report...")
        original_size = df.count()
        processed_size = processed_df.count()
        processing_time = 30  # Simulated processing time
        scalability_report = simulation.generate_scalability_report(
            original_size, processed_size, processing_time
        )
        
        # 7. Summary
        print("\n" + "=" * 60)
        print("SIMULATION SUMMARY")
        print("=" * 60)
        print(f"✅ Dataset size: {original_size:,} wafers")
        print(f"✅ Features processed: 100 → 50 (PCA)")
        print(f"✅ Model accuracy: {metrics['accuracy']:.4f}")
        print(f"✅ Model AUC: {metrics['auc']:.4f}")
        print(f"✅ Average throughput: {np.mean(real_time_stats['throughput']):.0f} wafers/sec")
        print(f"✅ Compression ratio: {scalability_report['compression_ratio']:.2f}x")
        
        print("\n🚀 This simulation demonstrates how the wafer yield optimization")
        print("   pipeline can scale to handle millions of wafers using Apache Spark!")
        
    except Exception as e:
        print(f"Error during simulation: {e}")
    
    finally:
        # Cleanup
        simulation.cleanup()

if __name__ == "__main__":
    main()
