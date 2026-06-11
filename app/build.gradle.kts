plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "com.example.asrdemo"
    compileSdk {
        version = release(36) {
            minorApiLevel = 1
        }
    }

    defaultConfig {
        applicationId = "com.example.asrdemo"
        minSdk = 24
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        ndk {
            // Mi 11 Pro (Snapdragon 888) 为 arm64；只打这一个 ABI 以缩小 APK
            abiFilters += "arm64-v8a"
        }
    }

    buildTypes {
        release {
            optimization {
                enable = false
            }
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    buildFeatures {
        compose = true
    }

    // 模型文件较大，禁止 aapt 压缩 onnx，避免安装时解压开销
    androidResources {
        noCompress += listOf("onnx", "txt")
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.activity.compose)
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.material3)

    // sherpa-onnx 官方预编译 AAR（含 libsherpa-onnx-jni.so 与 Kotlin API）
    // 来源: https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.2/sherpa-onnx-1.13.2.aar
    implementation(files("libs/sherpa-onnx-1.13.2.aar"))
}
