gcloud auth login
gcloud config set compute/zone us-east1-c
gcloud config set project symmetric-card-157400
wget https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2016-01.csv 
gcloud compute copy-files yellow_tripdata_2016-01.csv  cluster-1-m:~/
hadoop fs -put yellow_tripdata_2016-01.csv
spark-shell
val text = sc.textFile("yellow_tripdata_2016-01.csv")

//Vendor ID
//tpep_pickup_datetime
//tpep_dropoff_datetime
//passenger_count  *** 0
// trip distance ***1
//pickup_longitude **2
//pickup_latitude   ***3
//RatecodeID
//store_and_fwd_flag
//dropoff_longitude ***4
//dropoff_latitude **5
//payment_type ***6
//fare_amount **7
//extra
//mta_tax
//tip_amount **8
//tolls_amount  
//improvement_surcharge
//total_amount 

val data_noh = text.mapPartitionsWithIndex{ (idx,iter) => if (idx == 0) iter.drop(1) else iter }
val data = data_noh.map(l => l.split(","))


// west -74.2635 east -73.7526 south 40.4856 north40.9596
//formula to create zone: ((p.long.toDouble*39.1466+2907.16353).toInt)*10 + (p.lai.toDouble*21.09705-854.1267).toInt
//case class Scheme(passenger_count: Double, trip_distance: Double,pickup_longitude:Double,pickup_latitude:Double,dropoff_longitude:Double,dropoff_latitude:Double,payment_type:Double,fare_amount:Double,tip_amount:Double,pickup_zone:Int,dropoff_zone:Int)
val box_data = data.filter(p=> p(5).toDouble< -73.7526 && p(5).toDouble> -74.2635 && p(9).toDouble< -73.7526 && p(9).toDouble> -74.2635 && p(6).toDouble< 40.9596 && p(6).toDouble>40.4856 && p(10).toDouble< 40.9596 && p(10).toDouble>40.4856)
val useful_data = box_data.map(p => Array(p(3).toDouble, p(4).toDouble,p(5).toDouble,p(6).toDouble,p(9).toDouble,p(10).toDouble,p(11).toDouble,p(12).toDouble,p(15).toDouble,(((p(5).toDouble)*39.1466+2907.16353)toInt) *10 + (((p(6).toDouble)*21.09705- 854.1267).toInt),((((p(9).toDouble)*39.1466+2907.16353)toInt) *10 + (((p(10).toDouble)*21.09705- 854.1267).toInt)))
// extrat tips
val tips_dropoff  =  useful_data.map(f => (f(10).toInt,f(8)))
val tips_dropoff_pairs = sc.parallelize(tips_dropoff.collect)
val combiner = (x: Double) => (1,x)
val merger = (x: (Int, Double),y: Double) => {
val (c,acc) = x
(c +1 , acc+y)
}

val mergeAndCombiner = (x1: (Int, Double), x2: (Int, Double)) => {
val (c1, acc1) = x1
val (c2, acc2) = x2
(c1+c2,acc1+acc2)
}
val average_tip = tips_dropoff_pairs.combineByKey(combiner,merger,mergeAndCombiner)
val getAvgFunction = (x: (Int, (Int, Double))) => {
val (identifier, (count, total)) = x
(identifier,total/count)
}
val averages_tip_final = average_tip.collectAsMap().map(getAvgFunction)
ListMap(averages_tip_final.toSeq.sortWith(_._2 > _._2):_*)

//most frequent pick-up location
val pick_up = useful_data.map(f => (f(9).toInt,1))
pick_up.countByKey()

// most frequent drop_off location
val drop_off = useful_data.map(f => (f(10).toInt,1))
val drop_off_result =  drop_off.countByKey()
ListMap(drop_off_result.toSeq.sortWith(_._2>_._2):_*)

//filter to 15
val frequent_zone = useful_data.filter(f => f(10).toInt==105)
val good_data = frequent_zone.filter(f => f(7)>0)
val tip_rate = good_data.map(f => f(8)/f(7))
tip_rate.histogram(10)
tip_rate.histogram(Array(0.0, 0.05,0.1, 0.15,0.2, 0.25,0.3,0.35,0.4,0.45,0.5))

//Logistic regression
// predict zero vs nonzero tip based
//pick up zone, drop off zone, trip distance, passenger count and payment type
def dummy(x:Double): Double = {
    if(x >  0) 1
    else 0
}

val model_data = useful_data.map( f => Array(f(9),f(10),f(1),f(0),f(6),dummy(f(8))))

import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors



val labeled_data = model_data.map(l => LabeledPoint(l(5),Vectors.dense(l.slice(0,5))))
labeled_data.cache

//Setup the model and run it on the LabeledPoint dataset
val model = new LogisticRegressionWithLBFGS().setNumClasses(2).run(labeled_data)
model.intercept
model.weights

val predictionAndLabels = labeled_data.map { case LabeledPoint(label, features) =>
val prediction = model.predict(features)
(prediction, label)
}

//the model defaults to a .5 threshold. Clear that
model.clearThreshold
//set up a metrics object
val metrics = new BinaryClassificationMetrics(predictionAndLabels)
//Get model precision for many threshold values
val precision = metrics.precisionByThreshold
//Note: Precision = TP/(TP+FP)
//Maximum precision threshold
precision.collectAsMap.maxBy(_._2)
//All precisions
precision.foreach { case (t, p) =>
println(s"Threshold: $t, Precision: $p")
}


//Get model recall
val recall = metrics.recallByThreshold
//Note: recall = TP/(TP + FN)
//max recall
recall.collectAsMap.maxBy(_._2)
//map the two to get a precision recall curve
val PRC = metrics.pr
//Note: Precision = number of correct results/total number of results
//Calculate the F-score
val f1Score = metrics.fMeasureByThreshold
//Note: 2 * (P * R)/(P + R)

//Calculate the ROC
val roc = metrics.roc
//x-axis: False Positive Rate (FP/(FP+TN))
//y-axis: True Positive Rate (TP/(TP+FN))
//Calculate the AUC (area under the curve)
val auROC = metrics.areaUnderROC
