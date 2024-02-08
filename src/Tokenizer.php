<?php

namespace Textualization\SentenceTransphormers;

class Tokenizer extends \Textualization\Ropherta\Tokenizer {
    
    public function __construct($useCache = true)
    {
        parent::__construct($useCache);
    }
    
    public function encode(string $text): array
    {
        $result = parent::encode($text);
        \array_unshift($result, 0);
        $result[] = 2;
        return $result;
    }

    public function count(string $text): int
    {
        return parent::count($text) + 2;
    }
}

