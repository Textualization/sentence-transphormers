<?php

namespace Textualization\SentenceTransphormers;

use \Textualization\Ropherta\XLMTokenizer;

class SentenceXLMRopherta extends \Textualization\Ropherta\RophertaModel {

    public XLMTokenizer $xlmTokenizer;

    public function __construct($model=null, $input_size=512)
    {
        $model = $model ?? Vendor::model();
        parent::__construct($model, $input_size, true);
        $this->xlmTokenizer = new XLMTokenizer();
    }

    public function _encode(array|string $text, float|bool $padding = false) : array
    {
        return $this->_encode_passage($text, $padding);
    }
    
    public function _encode_passage(array|string $text, float|bool $padding = false) : array
    {
        return $this->_encode_call("passage: $text", $padding);
    }
    
    public function _encode_call(array|string $text, float|bool $padding = false) : array
    {
        $tokens = $this->xlmTokenizer->encode($text);
        
        if(count($tokens) > $this->input_size) {
            $tokens = \array_slice($tokens, 0, $this->input_size);
        }
        $input=[];
        if(count($this->model->inputs()) > 1) {
            // has mask
            $mask = [];
            foreach($tokens as $tok){
                $mask[] = 1.0;
            }
            if($padding) {
                while(count($mask) < $this->input_size) {
                    $mask[] = 0.0;
                }
            }
            $input['attention_mask'] = [ $mask ];
        }
        if($padding) {
            while(count($tokens) < $this->input_size) {
                $tokens[] = $padding;
            }
        }
        $input[ 'input_ids' ] = [ $tokens ];
        $zeros = array_fill(0, count($tokens), 0);
        $input[ 'token_type_ids' ] = [ $zeros ];
        
        $full_output = $this->model->predict($input);
        $layer = $full_output['last_hidden_state'];

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

        return $pool;
    }

    public function _encode_query(array|string $text, float|bool $padding = false) : array
    {
        return $this->_encode_call("query: $text", $padding);
    }
    
}
