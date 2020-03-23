#include <jni.h>
#include "coregistration.h"
#include "coregistration_trainer.h"
#include "src/solver/solver.h"
#include "src/base/timer.h"

void Java_com_inventale_coregistration_survey_providers_fm_XlearnTrainer_train(JNIEnv *env, jobject object,
                                                                               jobjectArray argsArray) {
    jboolean isQuiet = isQuietCheck(env, object);
    if (isQuiet) std::cout.setstate(std::ios_base::failbit);
    Timer timer;
    timer.tic();
    int argsCount = env->GetArrayLength(argsArray);
    typedef char *pchar;
    auto *argv = new pchar[argsCount];
    for (int i = 0; i < argsCount; i++) {
        auto jStringArg = (jstring) (env->GetObjectArrayElement(argsArray, i));
        const char *rawString = env->GetStringUTFChars(jStringArg, nullptr);
        argv[i] = new char[strlen(rawString) + 1];
        strcpy(argv[i], rawString);
        env->ReleaseStringUTFChars(jStringArg, rawString);
    }
    xLearn::Solver solver;
    solver.SetTrain();
    solver.Initialize(argsCount, argv);
    solver.StartWork();
    solver.Clear();
    Color::print_info(StringPrintf("Total time cost: %.6f (sec)", timer.toc()), NOT_IMPORTANT_MSG);
    if (isQuiet) std::cout.clear();
}