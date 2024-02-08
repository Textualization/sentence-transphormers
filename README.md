# Compute RoBERTa sentence embeddings in PHP using ONNX framework.

This brings the power of Sentence Transformers to the PHP world.

## Installation

Add this project to your dependencies

```
composer require textualization/sentence-transphormers
composer update
```

Before using it, you will need to install the ONNX framework:

```
composer exec -- php -r "require 'vendor/autoload.php'; OnnxRuntime\Vendor::check();"
```

and download the Distill RoBERTa v1 ONNX model (this takes a while, the model is 362Mb in size):

```
composer exec -- php -r "require 'vendor/autoload.php'; Textualization\SentenceTransphormers\Vendor::check();"
```

## Computing embeddings

```php

$model = new SentenceRophertaModel();

$emb = $model->embeddings("Text");
```

Check `\Textualization\Ropherta\Distances` to check whether two embeddings are closer to each other.

## Model employed

The model being used is an ONNX export from [sentence-transformers/all-distilroberta-v1](https://huggingface.co/sentence-transformers/all-distilroberta-v1), hosted at HuggingFace Hub: [textualization/all-distilroberta-v1](https://huggingface.co/textualization/all-distilroberta-v1).


