package com.ferraz.facerecognition

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.camera.core.ImageCapture.OnImageCapturedCallback
import androidx.camera.core.ImageProxy
import androidx.camera.view.LifecycleCameraController
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.PhotoCamera
import androidx.compose.material3.Button
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.ferraz.facerecognition.ui.theme.FRTheme
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        if (!hasCameraPermission()) {
            ActivityCompat.requestPermissions(this, CAMERAX_PERMISSIONS, 0)
        }

        enableEdgeToEdge()
        setContent {
            FRTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    FaceRecognition(
                        modifier = Modifier.padding(innerPadding)
                    )
                }
            }
        }
    }

    private fun hasCameraPermission(): Boolean {
        return CAMERAX_PERMISSIONS.all {
            ContextCompat.checkSelfPermission(
                baseContext,
                it
            ) == android.content.pm.PackageManager.PERMISSION_GRANTED
        }
    }

    companion object {
        private val CAMERAX_PERMISSIONS = arrayOf(
            android.Manifest.permission.CAMERA
        )
    }
}

@Composable
private fun FaceRecognition(modifier: Modifier = Modifier) {
    val context = LocalContext.current
    val cameraController = remember {
        LifecycleCameraController(context).apply {
            setEnabledUseCases(
                LifecycleCameraController.IMAGE_CAPTURE
            )
        }
    }

    val highAccuracyOpts = remember {
        FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .build()
    }
    val faceDetector = remember {
        FaceDetection.getClient(highAccuracyOpts)
    }

    val faceClassifier = remember {
        TFLiteFaceClassifier.create(
            context.assets,
            "facenet.tflite"
        )
    }

    var registeredFace by remember { mutableStateOf<Pair<Bitmap, FloatArray>?>(null) }

    var compareFace by remember { mutableStateOf<Pair<Bitmap, FloatArray>?>(null) }

    var action by remember { mutableStateOf<((Bitmap, FloatArray) -> Unit)?>(null) }

    var distance by remember { mutableStateOf<Float?>(null) }

    LaunchedEffect(key1 = registeredFace, key2 = compareFace) {
        if (registeredFace != null && compareFace != null) {
            distance = faceClassifier.compareEmbeddings(registeredFace!!.second, compareFace!!.second)
        }
    }

    Box(
        modifier = modifier.fillMaxSize(),
        contentAlignment = Alignment.BottomCenter
    ) {
        action?.let { saveImage ->
            CaptureImage(
                cameraController = cameraController,
                context = context,
                faceDetector = faceDetector,
                faceClassifier = faceClassifier
            ) { bitmap, embedding ->
                saveImage(bitmap, embedding)
                action = null
            }
        } ?: run {
            Column(
                modifier = Modifier.fillMaxSize(),
                verticalArrangement = Arrangement.Center,
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                distance?.let {
                    Text(text = "Distance: $it")
                }
                registeredFace?.let { face ->
                    Text(text = "Registered Face")
                    Image(
                        modifier = Modifier.fillMaxWidth(0.5f),
                        bitmap = face.first.asImageBitmap(),
                        contentDescription = ""
                    )
                }
                compareFace?.let { face ->
                    Text(text = "Compare Face")
                    Image(
                        modifier = Modifier.fillMaxWidth(0.5f),
                        bitmap = face.first.asImageBitmap(),
                        contentDescription = ""
                    )
                }
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                Button(
                    enabled = registeredFace == null,
                    onClick = {
                        action = { bitmap, embedding ->
                            registeredFace = bitmap to embedding
                        }
                    }
                ) {
                    Text(text = "Register Face")
                }

                Button(
                    onClick = {
                        action = { bitmap, embedding ->
                            compareFace = bitmap to embedding
                        }
                    }
                ) {
                    Text(text = "Compare Face")
                }
            }
        }
    }
}

@Composable
private fun CaptureImage(
    cameraController: LifecycleCameraController,
    context: Context,
    faceDetector: FaceDetector,
    faceClassifier: FaceClassifier,
    saveImage: (Bitmap, FloatArray) -> Unit
) {
    CameraPreview(
        controller = cameraController,
        modifier = Modifier.fillMaxSize()
    )
    IconButton(
        modifier = Modifier.padding(bottom = 16.dp),
        onClick = {
            takePhoto(
                controller = cameraController,
                context = context
            ) { bitmap ->
                val image = InputImage.fromBitmap(bitmap, 0)
                faceDetector.process(image)
                    .addOnSuccessListener { faces ->
                        faces.firstOrNull()?.boundingBox?.let { bounds ->
                            Bitmap.createBitmap(
                                bitmap,
                                bounds.left,
                                bounds.top,
                                bounds.width(),
                                bounds.height()
                            ).also { croppedBitmap ->
                                val embedding = faceClassifier.getFaceEmbeddings(croppedBitmap)
                                saveImage(croppedBitmap, embedding)
                            }
                        }
                    }
            }
        }
    ) {
        Icon(
            imageVector = Icons.Default.PhotoCamera,
            contentDescription = null
        )
    }
}

private fun takePhoto(
    controller: LifecycleCameraController,
    context: Context,
    onPhotoTaken: (Bitmap) -> Unit
) {
    controller.takePicture(
        ContextCompat.getMainExecutor(context),
        object : OnImageCapturedCallback() {
            override fun onCaptureSuccess(image: ImageProxy) {
                super.onCaptureSuccess(image)
                val matrix = Matrix().apply {
                    postRotate(image.imageInfo.rotationDegrees.toFloat())
                    postScale(-1f, 1f)
                }
                val rotatedBitmap = Bitmap.createBitmap(
                    image.toBitmap(), 0, 0, image.width, image.height, matrix, true
                )
                onPhotoTaken(rotatedBitmap)
                image.close()
            }
        }
    )
}

@Preview
@Composable
private fun FaceRecognitionPreview() {
    FRTheme {
        FaceRecognition()
    }
}
