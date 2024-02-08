<?php

namespace Textualization\SentenceTransphormer;

class SentenceRopherta extends \Textualization\Ropherta\RophertaModel {

    public function __construct(array $params)
    {
        $model = "../sentence-transphormers/all-distilroberta-v1/model.onnx";
        $input_size = $params["input_size"] ?? 512;
        parent::__construct($model, $input_size);
        $this->tokenizer = new Tokenizer(); // this creates the tokenizer twice, oh well
    }

    public function _encode(array|string $text) : array
    {
        $full_output = parent::_encode($text);

        $layer = $full_output["token_embeddings"];

        // mean pool
        $pool = $layer[0][0];
        $len = count($pool);
        $wlen = count($layer[0]);
        for($i=1; $i<$wlen; $i++)
            for($j=0; $j<$len; $j++)
                $pool[$j] += $layer[0][$i][$j];
        $sum_of_sqrs = 0.0;
        for($j=0; $j<$len; $j++) {
            $pool[$j] /= $wlen;
            $sum_of_sqrs += $pool[$j]*$pool[$j];
        }
        
        // normalize
        $norm = sqrt($sum_of_sqrs);
        for($j=0; $j<$len; $j++) {
            $pool[$j] /= $norm;
        }

        #echo json_encode($pool)."\n";
        
        return $pool;
    }
}
