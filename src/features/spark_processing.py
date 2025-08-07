from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.sql.functions import when

import os, sys

# … tes vérifs JAVA_HOME/HADOOP_HOME …

# ✅ Force HADOOP_HOME avec des slashs AVANT la session
os.environ["HADOOP_HOME"] = r"C:/hadoop/hadoop-3.3.6"
os.environ["PATH"] = os.environ["HADOOP_HOME"] + r"/bin;" + os.environ.get("PATH","")

def create_spark_session():
    return (
        SparkSession.builder
        .appName("HIV_Classification_Preprocessing")
        .config(
            "spark.driver.extraJavaOptions",
            "-Dorg.apache.hadoop.io.native.lib.available=false "
            "-Dhadoop.home.dir=C:/hadoop/hadoop-3.3.6"
        )
        .config(
            "spark.executor.extraJavaOptions",
            "-Dorg.apache.hadoop.io.native.lib.available=false "
            "-Dhadoop.home.dir=C:/hadoop/hadoop-3.3.6"
        )
        # Forcer le FS Java pur + committer v2
        .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2")
        # ✅ Forcer un committer direct pour éviter mergePaths
        .config("spark.hadoop.mapreduce.outputcommitter.factory.scheme.file",
                "org.apache.hadoop.mapreduce.lib.output.DirectFileOutputCommitterFactory")
        .getOrCreate()
    )



def load_data(spark, input_path):
    """
    Loads a CSV file as a Spark DataFrame.
    """
    return spark.read.csv(input_path, header=True, inferSchema=True)

#Spark ML does not accept separate columns like MolWt, TPSA, etc.
#You first need to assemble them into a single column called features, which contains a numeric vector.

def assemble_features(df, feature_cols, output_col="features"):
    """
    Assembles a list of feature columns into a single Spark vector column.
    """
    assembler = VectorAssembler(inputCols=feature_cols, outputCol=output_col)
    df_vector = assembler.transform(df)
    return df_vector


def scale_features(df, input_col="features", output_col="scaledFeatures"):
    """
    Applies standard scaling to the features vector column.
    """
    scaler = StandardScaler(inputCol=input_col, outputCol=output_col, withMean=True, withStd=True)
    scaler_model = scaler.fit(df)
    df_scaled = scaler_model.transform(df)
    return df_scaled


def add_active_column(df, threshold=1000):
    """
    Creates a binary column 'active' from 'standard_value'.
    1 if standard_value < threshold, else 0.
    """
    return df.withColumn("active", when(col("standard_value") < threshold, 1).otherwise(0))


def prepare_final_dataset(spark,df, output_path):
    """
    Keeps only valid rows with active labels and saves scaled features and label to disk.
    """
    df_clean = df.filter(col("active").isNotNull())

    df_final = df_clean.select(
        col("scaledFeatures").alias("features"),
        col("active").cast("int").alias("label")
    )
    spark.conf.set("spark.hadoop.io.native.lib.available", "false")
    df_final.write.mode("overwrite").parquet(output_path)
    print(f"✅ Saved final dataset with {df_final.count()} rows to {output_path}")





if __name__ == "__main__":
    # 1. Start Spark
    spark = create_spark_session()

    spark.conf.set("spark.hadoop.hadoop.native.lib", "false")
    spark.conf.set("spark.hadoop.fs.hdfs.impl", "org.apache.hadoop.hdfs.DistributedFileSystem")
    spark.conf.set("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2")
    spark.conf.set("spark.hadoop.mapreduce.fileoutputcommitter.cleanup-failures.ignored", "true")

    

    # 2. Load CSV with descriptors and standard_value
    df = load_data(spark, "data/processed/hiv_ic50_featurized.csv")

    # 3. Create the binary target column 'active'
    df = add_active_column(df, threshold=1000)

    # 4. Define the list of input features
    feature_cols = ["MolWt", "TPSA", "NumRotatableBonds",
                    "NumHDonors", "NumHAcceptors", "NumAromaticRings", "LogP"]

    # 5. Assemble features into a single vector
    df_vector = assemble_features(df, feature_cols)

    # 6. Apply standard scaling
    df_scaled = scale_features(df_vector)

    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)

    # 7. Filter, rename, and save the final dataset
    prepare_final_dataset(spark,df_scaled, "data/processed/hiv_prepared.parquet")


    # 8. Stop Spark
    spark.stop()
