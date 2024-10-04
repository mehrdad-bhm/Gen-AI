# Gen-AI
The primary objective of this project is to show the ability to fine-tune to enhance the
model performance in a specific task. Also, it will show how existing large language
models can be used in the data collection process to enhance relatively small models.
This will involve the use of a suitable dataset for fine-tuning a model for code generation,
the choice of a suitable small model to be fine-tuned on the Colab notebook, the design
of the experiment setup and evaluation matrices and choosing the correct prompt for
generation of the synthesised data.

## Step By Step Implementation:
  * Load the dataset benchmark and split it in the designed ratio.
  * Load your selected pre-trained model [Model A].
  * Test model A on the testing dataset using the selected evaluation metric.
  * Fine-tune model A on the training dataset [Model B].
  * Test model B on the testing dataset using the selected evaluation metric.
  * Use the designed prompt to generate a new synthesised dataset that has the
  nature and 3 times the size of the training dataset using the supported AWS
  model.
  * Fine-tune model A on the new synthesised dataset [Model C].
  * Test model C on the testing dataset using the selected evaluation metric.
  * Combine the training dataset and the synthesised dataset and shuffle them with
  suitable seeds.
  * Fine-tune model A on the new Combined dataset [Model D].
  * Test model D on the testing dataset using the selected evaluation metric.
  * Plot the right visualisation to show all models' performance.


## Dataset: mbpp | Mostly Basic Python Problems
  * Language: English - Python code
  * Data Splits: train, evaluation, test, prompt

## Model: croissantllm/CroissantLLMChat-v0.1
  * Size: 1.3 Billion parameters
  * Languages: Primarily French and English and suitable with code
  * Task: Text2Text Generation
  * Training Data: 3 Trillion tokens (3T) of text
  * Applications: Machine translation, Text generation
  
## Evaluation Result of Fine-tuned Models:
  * model A  —> BLEU score = 0.229
  * model B  —> BLEU score = 0.267
  * model C  —> BLEU score = 0.248
  * model D  —> BLEU score = 0.271
