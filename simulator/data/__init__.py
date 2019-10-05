import os
import xacro


def _gen_xacro(name, file='main', urdf_dir="urdf"):
    root = os.path.dirname(os.path.relpath(__file__))
    main_file = os.path.join(root, name, urdf_dir, '%s.xacro' % (file, ))

    out_dir = os.path.join(root, name, "gen")
    out_file = os.path.join(out_dir, "%s.urdf" % (file, ))

    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass

    with open(out_file, mode="w") as out:
        xacro.process_file(main_file).writexml(out, indent="\t", addindent="\t", newl="\n")

    return os.path.abspath(out_file)


def abb_irb120():
    return _gen_xacro("irb120")
