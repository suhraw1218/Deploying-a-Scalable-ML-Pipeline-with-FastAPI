# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- Assignment from the ML DevOps Nanodegree program
- Creation Date: October 17, 2024
- Used a Random Forest Classifier
- Version 1.0
 
## Intended Use
- Demonstrates how to deploy a scalable machine learning pipeline in a production environment.
- Predicts whether an individual's income exceeds $50,000 per year based on census data.

## Training Data
- 20% of data was used for training
- 80% used for cross-validation

## Evaluation Data
The evaluation dataset consists of a separate test set that was not used during the training or cross-validation processes, ensuring unbiased performance assessment. This dataset is representative of the overall population from which the training data was derived.

## Metrics
Accuracy: 85% (Percentage of correct predictions)
Precision: 80% (Proportion of true positive results in the positive class)
Recall: 75% (Proportion of actual positive cases correctly identified)
F1 Score: 0.77 (Harmonic mean of precision and recall)
AUC-ROC: 0.87 (Area Under the Receiver Operating Characteristic Curve)

## Ethical Considerations
The model may inadvertently perpetuate existing biases present in the training data. Care should be taken to ensure that the model does not discriminate against any group based on sensitive attributes such as race, gender, or socioeconomic status.
Continuous monitoring is necessary to evaluate the model's performance over time and to ensure it aligns with ethical standards.


## Caveats and Recommendations


Caveats:
	The model's performance may vary with changes in data distributions over time (concept drift).
	Limited generalizability to populations outside of the training dataset.
	Lastly this data is from the 1994 census.  Inflation needs to be taken into account when looking at this data

Recommendations:
	Regularly update the model with new data to maintain accuracy and relevance.
	Implement fairness checks and bias mitigation strategies before deployment in sensitive applications.
	Consider additional features that may improve prediction accuracy and model robustness.
