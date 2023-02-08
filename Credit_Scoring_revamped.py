#!/usr/bin/env python
# coding: utf-8

# Importing libraries
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import os
import xml.etree.ElementTree as ET
from cmlbootstrap import CMLBootstrap 

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator


def vectorize_cat_cols(df, cat_str_cols):
    idx_cols = [column+"_index" for column in cat_str_cols]
    enc_cols = [idx_column+"_vec" for idx_column in idx_cols]
    indexers = StringIndexer(inputCols=cat_str_cols, outputCols=idx_cols)
    encoders = OneHotEncoder(inputCols=idx_cols, outputCols=enc_cols, dropLast=False)
    pipeline = Pipeline(stages=[indexers, encoders])
    df_r = pipeline.fit(df).transform(df)
    df_r = df_r.drop(*idx_cols)
    df_r.select(*cat_str_cols, *enc_cols).toPandas()
    return df_r



# Standard Scaler
def make_pipeline_numeric(spark_df):
    stages= []

    scale_cols = spark_df.columns
    scale_cols.remove('is_default')

    #Assembling mixed data type transformations:
    assembler = VectorAssembler(inputCols=scale_cols, outputCol="features")
    stages += [assembler]

    #Standard scaler
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                        withStd=True, withMean=True)
    stages += [scaler]

    #Creating and running the pipeline:
    pipeline = Pipeline(stages=stages)
    pipelineModel = pipeline.fit(spark_df)
    out_df = pipelineModel.transform(spark_df)

    return out_df


def extract(row):
    return tuple(row.scaledFeatures.toArray().tolist()) + (row.is_default, )


def make_pipeline(spark_df, label_col):        
     
    #for c in spark_df.columns:
    #    spark_df = spark_df.withColumn(c, spark_df[c].cast("float"))
    
    stages= []

    #cols = ['acc_now_delinq', 'acc_open_past_24mths', 'annual_inc', 'avg_cur_bal', 'funded_amnt']
    cols = spark_df.columns
    
    #spark_df = spark_df.withColumn('acc_now_delinq',F.abs(spark_df['acc_now_delinq']))
    #spark_df = spark_df.withColumn('acc_open_past_24mths',F.abs(spark_df['acc_open_past_24mths']))
    #spark_df = spark_df.withColumn('annual_inc',F.abs(spark_df['annual_inc']))
    #spark_df = spark_df.withColumn('avg_cur_bal',F.abs(spark_df['avg_cur_bal']))
    #spark_df = spark_df.withColumn('funded_amnt',F.abs(spark_df['funded_amnt']))
    
    #Assembling mixed data type transformations:
    assembler = VectorAssembler(inputCols=cols, outputCol="features")
    stages += [assembler]    
    
    #Scaling features
    #scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True)
    #stages += [scaler]
    
    
    #RF Classifier
    #rf = LinearSVC(featuresCol='features', labelCol='is_default')
    rf = RandomForestClassifier(labelCol=label_col, featuresCol='features', numTrees=10)
    stages += [rf]
    
    #Creating and running the pipeline:
    pipeline = Pipeline(stages=stages)
    pipelineModel = pipeline.fit(spark_df)

    return pipelineModel


if __name__ == "__main__":
    # Setting STORAGE
    # Set the setup variables needed by CMLBootstrap
    HOST = os.getenv("CDSW_API_URL").split(":")[0] + "://" + os.getenv("CDSW_DOMAIN")
    USERNAME = os.getenv("CDSW_PROJECT_URL").split("/")[6]  # args.username  # "vdibia"
    API_KEY = os.getenv("CDSW_API_KEY")
    PROJECT_NAME = os.getenv("CDSW_PROJECT")

    # Instantiate API Wrapper
    cml = CMLBootstrap(HOST, USERNAME, API_KEY, PROJECT_NAME)

    try:
        storage = os.environ["STORAGE"]
    except:
        if os.path.exists("/etc/hadoop/conf/hive-site.xml"):
            tree = ET.parse("/etc/hadoop/conf/hive-site.xml")
            root = tree.getroot()
            for prop in root.findall("property"):
                if prop.find("name").text == "hive.metastore.warehouse.dir":
                    storage = (
                            prop.find("value").text.split("/")[0]
                            + "//"
                            + prop.find("value").text.split("/")[2]
                    )
        else:
            storage = "/user/" + os.getenv("HADOOP_USER_NAME")
        storage_environment_params = {"STORAGE": storage}
        storage_environment = cml.create_environment_variable(storage_environment_params)
        os.environ["STORAGE"] = storage

    print(os.environ["STORAGE"])

    # Creating Spark session
    spark = SparkSession \
        .builder \
        .appName("Credit Scoring") \
        .config("spark.hadoop.fs.s3a.s3guard.ddb.region", "us-east-2") \
        .config("spark.yarn.access.hadoopFileSystems", os.environ["STORAGE"]) \
        .getOrCreate()

    # Read source data
    df = spark.read.option('inferschema', 'true').csv(
        os.environ["STORAGE"] + "/credit_demo_sv/LoanStats_2015_original.csv",
        header=True,
        sep=',',
        nullValue='NA'
    )

    # Count number of nulls for each column:
    df_pd_null_pct = df.select([((F.count(F.when(F.isnan(c) | F.col(c).isNull(), c)) / df.count()) * 100).alias(c) for c in df.columns])

    null_cols = df_pd_null_pct.toPandas().T[(df_pd_null_pct.toPandas().T > 1).any(axis=1)].index

    # Dropping NULL columns
    df_non_null = df.drop(*null_cols)

    cat_cols = [item[0] for item in df_non_null.dtypes if item[1].startswith('string')]
    num_cols = [item[0] for item in df_non_null.dtypes if item[1].startswith('in') or item[1].startswith('dou')]
    num_features, cat_features = num_cols, cat_cols

    df_non_null = df_non_null.dropna()

    df_labelled = df_non_null.withColumn("is_default", F.when(
        (df_non_null["loan_status"] == "Charged Off") | (df_non_null["loan_status"] == "Default"), 1).otherwise(0))

    # Drop unwanted columns
    drop_str_cols = ['addr_state', 'desc', 'earliest_cr_line', 'emp_length', 'emp_title', 'id', 'issue_d',
                     'loan_status', 'sub_grade', 'term', 'title', 'verification_status', 'zip_code']
    select_str_cat_cols = ['application_type', 'grade', 'home_ownership', 'initial_list_status', 'purpose']

    df_vec = vectorize_cat_cols(df_labelled, select_str_cat_cols)
    df_vec = df_vec.withColumn("int_rate", F.expr("substring(int_rate, 1, length(int_rate)-1)").cast('double'))
    df_vec = df_vec.withColumn("revol_util", F.expr("substring(revol_util, 1, length(revol_util)-1)").cast('double'))

    drop_cols = drop_str_cols
    drop_cols.append('loan_status')

    df_vec = df_vec.drop(*drop_cols)
    df_vec = df_vec.drop(*select_str_cat_cols)

    df_res = df_vec.select(*num_cols, 'is_default')
    df_model = make_pipeline_numeric(df_res)

    df_reb = df_model.select('scaledFeatures', 'is_default')
    df_reb_table = df_reb.rdd.map(extract).toDF([*num_cols, 'is_default'])

    df_base = df_reb_table
    train = df_base.sampleBy("is_default", fractions={0: 0.6, 1: 0.6}, seed=10)
    test = df_base.subtract(train)

    pipelineModel = make_pipeline(train, 'is_default')

    df_model_pred = pipelineModel.transform(test)

    evaluator = BinaryClassificationEvaluator(labelCol="is_default", rawPredictionCol="prediction")

    auroc = evaluator.evaluate(df_model_pred, {evaluator.metricName: "areaUnderROC"})
    auprc = evaluator.evaluate(df_model_pred, {evaluator.metricName: "areaUnderPR"})

    print("Area under ROC Curve: {:.4f}".format(auroc))
    print("Area under PR Curve: {:.4f}".format(auprc))

    y_true = df_model_pred.select(['is_default']).collect()
    y_pred = df_model_pred.select(['prediction']).collect()

    from sklearn.metrics import classification_report, confusion_matrix

    print(classification_report(y_true, y_pred))

    if auroc > 0.7:
        pipelineModel.write().overwrite().save(os.environ["STORAGE"] + "/svadivel/pipeline/baseline/")


    spark.stop()
