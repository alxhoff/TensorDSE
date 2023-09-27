from mltk.core import summarize_model
import wget

model_name = wget.download("https://github.com/alxhoff/TensorDSE/raw/master/resources/models/example_models/MNIST_full_quanitization.tflite")

summary = summarize_model(model_name, tflite=True)

print(summary)