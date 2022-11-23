from . import aistpp

def test_aistpp_kpoint_shape():
    time, joint, dof = aistpp.kpoint.DataList()[0].shape
    assert dof == 3 and joint == 17

def test_aistpp_smpl_shape():
    data = aistpp.smpl.DataList()
    time, joint, dof = data[0].shape
    try:
        assert dof == 3 and joint == 24
    except AssertionError as e:
        print(data[0].shape)
        raise e

    data = aistpp.smpl.DataList(include_rootp=True)
    p, root_p = data[0]
    
    # time dimension equals
    assert p.shape[0] == root_p.shape[0]


if __name__ == "__main__":
    for f in filter(lambda v:v.startswith("test"), dir()):
        eval(f"{f}()")