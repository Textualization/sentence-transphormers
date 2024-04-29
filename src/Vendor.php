<?php

namespace Textualization\SentenceTransphormers;

class Vendor {

    public static function model() {
        return self::libDir() . '/' . "model.onnx";
    }
    
    public static function check($multilingual=false) {
        $dest = self::model();

        $name = "all-distillroberta-v1";        
        if($multilingual) {
            $name = "multilingual-e5-small";
        }
        
        if(file_exists($dest)) {
            echo "✔ $name ONNX Model found\n";
            return;
        }
        
        $dir = self::libDir();
        if (!file_exists($dir)) {
            mkdir($dir);
        }

        echo "Downloading $name ONNX Model...\n";

        $url="https://huggingface.co/textualization/all-distilroberta-v1/resolve/60354dda2fe60375dd9c92115e72cc4788318783/model.onnx";
        if($multilingual){
            $url="https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/onnx/model.onnx";
        }
        
        $ch = curl_init();
        curl_setopt($ch, CURLOPT_URL, $url);
        curl_setopt($ch, CURLOPT_VERBOSE, false);
        curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
        curl_setopt($ch, CURLOPT_FOLLOWLOCATION, true);
        curl_setopt($ch, CURLOPT_HEADER, 0);
        $contents = curl_exec($ch);
        curl_close($ch);
        
        $checksum = hash('sha256', $contents);
        if($multilingual){
            if($checksum != "ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665") {
                throw new \Exception("Bad checksum: $checksum");
            }
        } elseif($checksum != "b4ecabee58ecc1c6aac890031c82132064c3e08aae2d50cffb5d04ba0f0a5d9c") {
            throw new \Exception("Bad checksum: $checksum");
        }
        file_put_contents($dest, $contents);
        
        echo "✔ Success\n";        
    }

    private static function libDir() {
        return __DIR__ . '/../lib';
    }
}
    
