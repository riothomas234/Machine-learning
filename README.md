This is a file containing machine learning projects.

[Makemore2](makemore part2.ipynb) implements a character-level neural network language model using PyTorch to generate names. The model learns character embeddings and predicts the next character in a sequence given a fixed-length context (3 characters). It is trained on a dataset of names, split into training, validation, and test sets (80/10/10).

[Makemore3](makemore3.ipynb) introduces trainable batch norm layers after each hidden layer to stabilize learning and reduce internal covariate shift. We track activations, gradient distributions, and update/data ratios to monitor network health, something not present in Make More 2. Weight scaling and output layer adjustments reduce saturation and overconfidence in predictions.
