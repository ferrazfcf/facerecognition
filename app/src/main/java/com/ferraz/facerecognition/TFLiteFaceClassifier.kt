package com.ferraz.facerecognition

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.sqrt

class TFLiteFaceClassifier private constructor(
    private val assetManager: AssetManager,
    private val modelFilename: String,
    private val inputSize: Int,
    private val outputSize: Int,
    private val isModelQuantized: Boolean
) : FaceClassifier {

    private val numBytesPerChannel: Int by lazy {
        if (isModelQuantized) 1 else 4
    }

    private val imgData: ByteBuffer by lazy {
        ByteBuffer.allocateDirect(
            1 * inputSize * inputSize * 3 * numBytesPerChannel
        ).order(ByteOrder.nativeOrder())
    }

    private val intValues: IntArray by lazy {
        IntArray(inputSize * inputSize)
    }

    private val interpreter: Interpreter by lazy {
        Interpreter(loadModelFile(assetManager, modelFilename).asReadOnlyBuffer())
    }

    override fun compareEmbeddings(emb1: FloatArray, emb2: FloatArray): Float {
        var distance = 0f
        for (i in emb1.indices) {
            val diff = emb1[i] - emb2[i]
            distance += diff * diff
        }
        return sqrt(distance)
    }

    override fun getFaceEmbeddings(bitmap: Bitmap): FloatArray {
        bitmap.resizeBitmapWithPadding().run {
            getPixels(intValues, 0, width, 0, 0, width, height)
        }
        imgData.rewind()
        for (i in 0 until inputSize) {
            for (j in 0 until inputSize) {
                val pixelValue = intValues[i * inputSize + j]
                if (isModelQuantized) {
                    imgData.put(((pixelValue shr 16) and 0xFF).toByte())
                    imgData.put(((pixelValue shr 8) and 0xFF).toByte())
                    imgData.put((pixelValue and 0xFF).toByte())
                } else {
                    imgData.putFloat((((pixelValue shr 16) and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                    imgData.putFloat((((pixelValue shr 8) and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                    imgData.putFloat(((pixelValue and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                }
            }
        }
        val output = Array(1) { FloatArray(outputSize) }

        interpreter.run(imgData, output)

        return output[0]
    }

    private fun Bitmap.resizeBitmapWithPadding(): Bitmap {
        val width = this.width
        val height = this.height
        val ratioBitmap = width.toFloat() / height.toFloat()
        val ratioMax = inputSize.toFloat() / inputSize.toFloat()

        var newWidth = inputSize
        var newHeight = inputSize
        if (ratioMax > ratioBitmap) {
            newWidth = (inputSize.toFloat() * ratioBitmap).toInt()
        } else {
            newHeight = (inputSize.toFloat() / ratioBitmap).toInt()
        }

        val resizedBitmap = Bitmap.createScaledBitmap(this, newWidth, newHeight, true)
        val result = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(result)
        canvas.drawColor(Color.GREEN)
        canvas.drawBitmap(resizedBitmap, (inputSize - newWidth) / 2f, (inputSize- newHeight) / 2f, null)
        return result
    }

    companion object {
        private const val IMAGE_MEAN = 128.0f
        private const val IMAGE_STD = 128.0f

        @Throws(IOException::class)
        private fun loadModelFile(assets: AssetManager, modelFilename: String): MappedByteBuffer {
            val fileDescriptor = assets.openFd(modelFilename)
            val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
            val fileChannel = inputStream.channel
            val startOffset = fileDescriptor.startOffset
            val declaredLength = fileDescriptor.declaredLength
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        }

        @Throws(IOException::class)
        fun create(
            assetManager: AssetManager,
            modelFilename: String,
            inputSize: Int = 160,
            outputSize: Int = 512,
            isModelQuantized: Boolean = false
        ) : FaceClassifier {
            return TFLiteFaceClassifier(assetManager, modelFilename, inputSize, outputSize, isModelQuantized)
        }
    }
}
