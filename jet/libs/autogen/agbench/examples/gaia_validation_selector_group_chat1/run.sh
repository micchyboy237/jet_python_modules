
echo RUN.SH STARTING !
export AUTOGEN_TESTBED_SETTING="Native"
echo "agbench version: 0.0.1a1" > timestamp.txt
export PYTHONPATH="/Users/jethroestrada/.pyenv/versions/3.12.9/lib/python3.12/site-packages:/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet:/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/shared_modules/shared:$PYTHONPATH"
if [ -f global_init.sh ] ; then
    . ./global_init.sh
fi
if [ -f scenario_init.sh ] ; then
    . ./scenario_init.sh
fi
echo SCENARIO.PY STARTING !
start_time=$(date +%s)
timeout --preserve-status --kill-after 7230s 7200s python scenario.py
end_time=$(date +%s)
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "SCENARIO.PY EXITED WITH CODE: $EXIT_CODE !"
else
    echo "SCENARIO.PY COMPLETE !"
fi
elapsed_time=$((end_time - start_time))
echo "SCENARIO.PY RUNTIME: $elapsed_time !"
if [ -d .cache ] ; then
    rm -Rf .cache
fi
if [ -d __pycache__ ] ; then
    rm -Rf __pycache__
fi
if [ -f scenario_finalize.sh ] ; then
    . ./scenario_finalize.sh
fi
if [ -f global_finalize.sh ] ; then
    . ./global_finalize.sh
fi
echo RUN.SH COMPLETE !
