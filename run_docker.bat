set APP_PATH=%cd%

docker stop single_word_gmmhmm_run
docker rm single_word_gmmhmm_run
docker run -it  -d  ^
    -p 0.0.0.0:3332:22  ^
    -p 0.0.0.0:7006:8888  ^
    -v %APP_PATH%:/opt/single_word_asr_gmm_hmm  ^
    --name single_word_gmmhmm_run  ^
    single_word_gmmhmm