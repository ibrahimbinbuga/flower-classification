import pandas as pd
import ace_tools as tools

custom_cnn_accuracy = 61.03
custom_cnn_precision = 0.6516
custom_cnn_recall = 0.6103
custom_cnn_f1 = 0.5941
custom_cnn_train_time = 1385

vgg16_feature_accuracy = 90.64
vgg16_feature_precision = 0.9082
vgg16_feature_recall = 0.9064
vgg16_feature_f1 = 0.9067
vgg16_feature_train_time = 3286

vgg16_finetuned_accuracy = 95.85
vgg16_finetuned_precision = 0.9586
vgg16_finetuned_recall = 0.9585
vgg16_finetuned_f1 = 0.9585
vgg16_finetuned_train_time = 7133

model_results = {
    'Model': ['Custom CNN', 'VGG16 Feature Extractor', 'VGG16 Fine-Tuned'],
    'Accuracy': [custom_cnn_accuracy, vgg16_feature_accuracy, vgg16_finetuned_accuracy],
    'Precision': [custom_cnn_precision, vgg16_feature_precision, vgg16_finetuned_precision],
    'Recall': [custom_cnn_recall, vgg16_feature_recall, vgg16_finetuned_recall],
    'F1-score': [custom_cnn_f1, vgg16_feature_f1, vgg16_finetuned_f1],
    'TrainTime (in s)': [custom_cnn_train_time, vgg16_feature_train_time, vgg16_finetuned_train_time]
}

df = pd.DataFrame(model_results)

tools.display_dataframe_to_user(name="Model Performance Comparison", dataframe=df)
