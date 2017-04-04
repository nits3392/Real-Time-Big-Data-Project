import org.apache.spark.rdd._
import scala.collection.JavaConverters._
import au.com.bytecode.opencsv.CSVReader

import java.io._
import org.joda.time._
import org.joda.time.format._

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD

case class AirDelayRecord(year: String,
                    month: String,
                    dayOfMonth: String,
                    dayOfWeek: String,
                    crsDepTime: String,
                    depDelay: String,
                    origin: String,
                    distance: String,
                    cancelled: String,
                    carrier: String) {

    def del_features: (String, Array[Double]) = {
      val values = Array(
        depDelay.toDouble,
        month.toDouble,
        dayOfMonth.toDouble,
        dayOfWeek.toDouble,
        get_hour(crsDepTime).toDouble,
        distance.toDouble
      )
      new Tuple2(to_date(year.toInt, month.toInt, dayOfMonth.toInt), values)
    }

    def get_hour(depTime: String) : String = "%04d".format(depTime.toInt).take(2)
    def to_date(year: Int, month: Int, day: Int) = "%04d%02d%02d".format(year, month, day)
}
//  preprocessing step for a given file to fliter data for specific city and year
def delayPrep(infile: String): RDD[AirDelayRecord] = {
    val data = sc.textFile(infile)

    data.map { line =>
      val reader = new CSVReader(new StringReader(line))
      reader.readAll().asScala.toList.map(rec => AirDelayRecord(rec(0),rec(1),rec(2),rec(3),rec(5),rec(15),rec(16),rec(18),rec(21),rec(8)))
    }.map(list => list(0))
    .filter(rec => rec.year != "Year")
    .filter(rec => rec.cancelled == "0")
    .filter(rec => rec.origin == "JFK")
}

val data_2007 = delayPrep("2007.csv").map(rec => rec.del_features._2)
val data_2008 = delayPrep("2008.csv").map(rec => rec.del_features._2)
data_2007.take(5).map(x => x mkString ",").foreach(println)


// removing delays below 15 mins and considering only delays above 15 mins. Delay is labelled as 1 for dela>15 else 0)
def parseData(vals: Array[Double]): LabeledPoint = {
  LabeledPoint(if (vals(0)>=15) 1.0 else 0.0, Vectors.dense(vals.drop(1)))
}

// Prepare training set
val pTrainData = data_2007.map(parseData)
pTrainData.cache
val stdScaler = new StandardScaler(withMean = true, withStd = true).fit(pTrainData.map(x => x.features))
val scaledTData = pTrainData.map(x => LabeledPoint(x.label, stdScaler.transform(Vectors.dense(x.features.toArray))))
scaledTData.cache

// Prepare test/validation set
val parsedTestData = data_2008.map(parseData)
parsedTestData.cache
val scaledTestData = parsedTestData.map(x => LabeledPoint(x.label, stdScaler.transform(Vectors.dense(x.features.toArray))))
scaledTestData.cache

scaledTData.take(3).map(x => (x.label, x.features)).foreach(println)


// Function to compute evaluation metrics
def eval_metrics(PredLabels: RDD[(Double, Double)]) : Tuple2[Array[Double], Array[Double]] = {
    val tp = PredLabels.filter(r => r._1==1 && r._2==1).count.toDouble
    val tn = PredLabels.filter(r => r._1==0 && r._2==0).count.toDouble
    val fp = PredLabels.filter(r => r._1==1 && r._2==0).count.toDouble
    val fn = PredLabels.filter(r => r._1==0 && r._2==1).count.toDouble

    val precision = tp / (tp+fp)
    val recall = tp / (tp+fn)
    val F_measure = 2*precision*recall / (precision+recall)
    val accuracy = (tp+tn) / (tp+tn+fp+fn)
    new Tuple2(Array(tp, tn, fp, fn), Array(precision, recall, F_measure, accuracy))
}


// Build the Decision Tree model
val numClasses = 2
val categoricalFeaturesInfo = Map[Int, Int]()
val impurity = "gini"
val maxDepth = 10
val maxBins = 100
val model_dt = DecisionTree.trainClassifier(pTrainData, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

// Predict
val PredLabels_dt = parsedTestData.map { point =>
    val pred = model_dt.predict(point.features)
    (pred, point.label)
}
val m_dt = eval_metrics(PredLabels_dt)._2
println("precision = %.2f, recall = %.2f, F1 = %.2f, accuracy = %.2f".format(m_dt(0), m_dt(1), m_dt(2), m_dt(3)))



// Build the SVM model
val classifier = new SVMWithSGD()
classifier.optimizer.setNumIterations(100)
classifier.setRegParam(1.0)
classifier.setStepSize(1.0)
val model_svm = classifier.run(scaledTrainData)

// Predict
val labelsAndPreds_svm = scaledTestData.map { point =>
        val pred = model_svm.predict(point.features)
        (pred, point.label)
}
val m_svm = eval_metrics(labelsAndPreds_svm)._2
println("precision = %.2f, recall = %.2f, F1 = %.2f, accuracy = %.2f".format(m_svm(0), m_svm(1), m_svm(2), m_svm(3)))



// Build the Logistic Regression model
val model_lr = LogisticRegressionWithSGD.train(scaledTrainData, numIterations=100)

// Predict
val labelsAndPreds_lr = scaledTestData.map { point =>
val pred = model_lr.predict(point.features)
    (pred, point.label)
}
val m_lr = eval_metrics(labelsAndPreds_lr)._2
println("precision = %.2f, recall = %.2f, F1 = %.2f, accuracy = %.2f".format(m_lr(0), m_lr(1), m_lr(2), m_lr(3))

