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
        return _encode_passage($text, $padding);
    }
    
    public function _encode_passage(array|string $text, float|bool $padding = false) : array
    {
        return _encode_call("passage: $text", $padding);
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
        
        return $this->model->predict($input);
    }

    public function _encode_query(array|string $text, float|bool $padding = false) : array
    {
        return $this->_encode_call("query: $text", $padding);
    }
    
}
