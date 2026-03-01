#
# system_utilities.mojo
# mosaic
#
# Created by Christian Bator on 02/02/2025
#

from os import abort
from sys import simdwidthof
from sys.info import CompilationTarget


@parameter
fn optimal_simd_width[dtype: DType]() -> Int:
    @parameter
    if CompilationTarget.is_macos():
        return 4 * simdwidthof[dtype]()
    else:
        return 2 * simdwidthof[dtype]()


alias unroll_factor = 4


@parameter
fn dynamic_library_filepath(name: String) -> String:
    if CompilationTarget.is_linux():
        return name + ".so"
    elif CompilationTarget.is_macos():
        return name + ".dylib"
    else:
        return abort[String]("Unsupported os for dynamic library filepath determination")
