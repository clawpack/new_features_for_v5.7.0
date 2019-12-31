
Example with particle tracking, test case with flow around an island

This example runs only with a version of Clawpack that contains some
new commits in 4 subrepositories.

To clone a new version of clawpack with the proper things checked out:

    git clone http://github.com/rjleveque/clawpack clawpack_lagrangian_gauges
    cd clawpack_lagrangian_gauges
    git submodule init
    git submodule update

and then set $CLAW to point to this version,
and also $PYTHONPATH, or else use `pip install -e .`

