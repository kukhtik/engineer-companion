# ProGuard rules for Engineer Companion
-keepclassmembers class com.varian.engcomp.LlamaEngine {
    native <methods>;
}
-keep class com.chaquo.python.** { *; }
-dontwarn com.chaquo.python.**
-dontwarn org.xmlpull.v1.**
-dontwarn com.google.auto.value.**
