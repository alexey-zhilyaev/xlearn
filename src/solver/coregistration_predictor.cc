#include <jni.h>
#include "coregistration.h"
#include "coregistration_predictor.h"
#include "src/solver/solver.h"
#include "src/base/timer.h"

const char *const INSTANCE_POINTER_FIELD = "xlearnInstancePointer";
const char *const RESULT_JAVA_CLASS = "com/inventale/coregistration/survey/providers/fm/PredictionResult";
jboolean isQuiet = false;

bool sortDescBySecond(const std::pair<int,float > &a, const std::pair<int,float> &b){
    return (a.second > b.second);
}

jfieldID getInstancePointerFieldId(JNIEnv * env, jobject obj){
    static jfieldID ptrFieldId = 0;
    if (!ptrFieldId){
        jclass clazz = env->GetObjectClass(obj);
        ptrFieldId = env->GetFieldID(clazz, INSTANCE_POINTER_FIELD, "J");
        env->DeleteLocalRef(clazz);
    }
    return ptrFieldId;
}

void Java_com_inventale_coregistration_survey_providers_fm_XlearnPredictor_init(JNIEnv *env, jobject object, jstring jmodel, jint threadsCount) {
    isQuiet = isQuietCheck(env, object);
    if (isQuiet) std::cout.setstate(std::ios_base::failbit);
    Color::print_action("Start Initializing");
    auto model = env->GetStringUTFChars(jmodel, nullptr);
    xLearn::HyperParam param;
    param.model_file = model;
    param.is_train = false;
    param.from_file = false;
    param.res_out = false;
    param.thread_number = threadsCount;
    xLearn::DMatrix matrix;
    param.test_dataset = &matrix;
    auto solver = new xLearn::Solver();
    solver->Initialize(param);
    env->SetLongField(object, getInstancePointerFieldId(env, object), (jlong) solver);
    Color::print_action("Finish Initializing");
}

void Java_com_inventale_coregistration_survey_providers_fm_XlearnPredictor_dispose(JNIEnv *env, jobject object) {
    auto solver = (xLearn::Solver*) env->GetLongField(object, getInstancePointerFieldId(env, object));
    solver->Clear();
    delete solver;
}

JNIEXPORT jobjectArray JNICALL
Java_com_inventale_coregistration_survey_providers_fm_XlearnPredictor_predict(JNIEnv *env, jobject object,
                                                                             jintArray tasks, jintArray keys,
                                                                             jintArray values, jint jtopSize) {
    Timer timer;
    timer.tic();
    Color::print_action("Reading input parameters ...");
    jint *taskArray = (env)->GetIntArrayElements(tasks, nullptr);
    jint *keysArray = (env)->GetIntArrayElements(keys, nullptr);
    jint *valuesArray = (env)->GetIntArrayElements(values, nullptr);
    jsize tasks_size = (env)->GetArrayLength(tasks);
    jsize facts_size = (env)->GetArrayLength(keys);
    Color::print_info(StringPrintf("Tasks amount: %i", tasks_size));
    Color::print_info(StringPrintf("Knowledge amount: %i", facts_size));

    Color::print_action("Generating prediction matrix ...");
    xLearn::DMatrix matrix;
    matrix.Reset();
    matrix.has_label = false;
    uint32 row_id = 0;
    for (size_t i = 0; i < tasks_size; i++) {
        matrix.AddRow();
        auto task_idx = (uint32) taskArray[i];
        matrix.AddNode(row_id, task_idx, 1);
        for (size_t j = 0; j < facts_size; j++) {
            uint32 fact_idx = keysArray[j];
            uint32 fact_value = valuesArray[j];
            matrix.AddNode(row_id, fact_idx, fact_value);
        }
        row_id++;
    }
    Color::print_info(StringPrintf("Prediction matrix was generated"));
    long solverPointer = env->GetLongField(object, getInstancePointerFieldId(env, object));
    auto solver = reinterpret_cast<xLearn::Solver*>(solverPointer);
    solver->InitializeDataset(&matrix);
    solver->StartWork();
    std::vector<float> result = solver->GetResult();
    Color::print_info(StringPrintf("Total predict time cost: %.3f (sec)", timer.toc()), false);

    // Selection of best tasks
    std::vector<std::pair<int, float> > taskToResult;
    for (int i = 0; i < result.size(); i++) {
        taskToResult.emplace_back(std::make_pair(taskArray[i], result[i]));
    }
    auto topSize = jtopSize > tasks_size ? tasks_size : jtopSize;
    std::partial_sort(taskToResult.begin(), taskToResult.begin() + topSize, taskToResult.end(), sortDescBySecond);

    auto cls = (env)->FindClass(RESULT_JAVA_CLASS);
    jobjectArray out = (env)->NewObjectArray(topSize, cls, nullptr);
    jmethodID constructor = (env)->GetMethodID(cls, "<init>", "(ID)V");
    if (nullptr == constructor) return nullptr;
    for (int i = 0; i < topSize; i++) {
        auto element =  taskToResult[i];
        auto newObject = (env)->NewObject(cls, constructor, element.first, element.second);
        (env)->SetObjectArrayElement(out, i, newObject);
    }
    // Release resources
    (env)->ReleaseIntArrayElements(tasks, taskArray, 0);
    (env)->ReleaseIntArrayElements(keys, keysArray, 0);
    (env)->ReleaseIntArrayElements(values, valuesArray, 0);
    return out;
}