<?php

namespace Textualization\SentenceTransphormers;

require 'vendor/autoload.php';

class Vendor {

    public static function model() {
        return self::libDir() . '/' . "model.onnx";
    }
    
    public static function check($event=null) {
        $dest = self::model();
        
        if(file_exists($dest)) {
            echo "✔ all-distillroberta-v1 ONNX Model found\n";
            return;
        }
        
        $dir = self::libDir();
        if (!file_exists($dir)) {
            mkdir($dir);
        }

        echo "Downloading all-distillroberta-v1 ONNX Model...\n";

        $url="https://huggingface.co/textualization/all-distilroberta-v1/resolve/60354dda2fe60375dd9c92115e72cc4788318783/model.onnx";
        $ch = curl_init();
        curl_setopt($ch, CURLOPT_URL, $url);
        curl_setopt($ch, CURLOPT_VERBOSE, true);
        curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
        curl_setopt($ch, CURLOPT_FOLLOWLOCATION, true);
        curl_setopt($ch, CURLOPT_HEADER, 0);
        $contents = curl_exec($ch);
        curl_close($ch);
        
        //$contents = file_get_contents($url, false, stream_context_create($options));

        $checksum = hash('sha256', $contents);
        if($checksum != "b4ecabee58ecc1c6aac890031c82132064c3e08aae2d50cffb5d04ba0f0a5d9c") {
            throw new \Exception("Bad checksum: $checksum");
        }
        file_put_contents($dest, $contents);
        
        echo "✔ Success\n";        
    }

    private static function libDir() {
        return __DIR__ . '/../lib';
    }
}
    
