
#include <jni.h>

const char *const IS_QUIET_FIELD = "isQuiet";

jboolean isQuietCheck(JNIEnv * env, jobject obj) {
    jclass clazz = env->GetObjectClass(obj);
    jfieldID fid = env->GetFieldID(clazz, IS_QUIET_FIELD, "Z");
    env->DeleteLocalRef(clazz);
    return env->GetBooleanField(obj, fid);
}