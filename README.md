import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.annotation.Since
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.mllib.regression.impl.GLMRegressionModel
import org.apache.spark.ml.feature.{VectorAssembler,VectorIndexer, StringIndexer, OneHotEncoder }
import org.apache.spark.sql.SparkSession
import org.apache.log4j._

Logger.getLogger("org").setLevel(Level.ERROR)

val spark = SparkSession.builder()getOrCreate()

val df= spark.read.option("header", "true").option("inferSchema","true").csv("movie_metadata.csv")

df.printSchema()
df.head(1)

val data= df.select(df("imdb_score").as("label"),$"director_facebook_likes",$"cast_total_facebook_likes",
                                        $"actor_1_facebook_likes",
                                        $"actor_2_facebook_likes",
                                        $"actor_3_facebook_likes",
                                        $"movie_facebook_likes",
                                        $"facenumber_in_poster",
                                        $"gross",
                                        $"budget")
val dataset = data.na.drop()

val assembler =( new VectorAssembler().setInputCols(Array("director_facebook_likes","cast_total_facebook_likes",
                                        "actor_1_facebook_likes",
                                        "actor_2_facebook_likes",
                                        "actor_3_facebook_likes",
                                        "movie_facebook_likes",
                                        "facenumber_in_poster",
                                        "gross",
                                        "budget")).setOutputCol("features"))

val Array(training, test)= dataset.randomSplit(Array(0.7,0.3),seed=664852)


import org.apache.spark.ml.Pipeline

var lr = new LinearRegression()
val pipeline= new Pipeline().setStages(Array(assembler,lr))
var model= pipeline.fit(training)
val results= model.transform(test)
import org.apache.spark.mllib.evaluation.MulticlassMetrics

val predictionAndLabels= results.select($"prediction", $"label").as[(Double, Double)].rdd


val metrics = new MulticlassMetrics(predictionAndLabels)

println("confusion Matrix :")

println(metrics.confusionMatrix)
