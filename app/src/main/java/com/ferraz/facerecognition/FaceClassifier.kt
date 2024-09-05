package com.ferraz.facerecognition

import android.graphics.Bitmap


interface FaceClassifier {
    fun getFaceEmbeddings(bitmap: Bitmap): FloatArray
    fun compareEmbeddings(emb1: FloatArray, emb2: FloatArray): Float
}
