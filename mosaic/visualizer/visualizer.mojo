#
# visualizer.mojo
# mosaic
#
# Created by Christian Bator on 12/14/2024
#

from os import abort
from sys.ffi import _Global, OwnedDLHandle, _get_dylib_function, c_int, c_char, c_float
from memory import UnsafePointer

from mosaic.utility import dynamic_library_filepath
from mosaic.image import Image, ImageSlice, ColorSpace

#
# Backend
#
comptime _libvisualizer = _Global["libvisualizer", _load_libvisualizer]()


fn _load_libvisualizer() -> OwnedDLHandle:
    try:
        return OwnedDLHandle(dynamic_library_filepath("libmosaic-visualizer"))
    except:
        abort()


#
# Visualizer
#
struct Visualizer:
    #
    # Fields
    #
    comptime display_dtype = DType.uint8

    #
    # ImageSlice
    #
    @staticmethod
    fn show(image_slice: ImageSlice, window_title: String):
        Self.show(image=image_slice.deep_copy(), window_title=window_title)

    #
    # Image
    #
    @staticmethod
    fn show[dtype: DType, color_space: ColorSpace, //](image: Image[color_space, dtype], window_title: String):
        @parameter
        if color_space.is_display_color_space() and dtype == Self.display_dtype:
            Self._show(image=image, window_title=window_title)
        elif color_space.is_display_color_space():
            Self._show(
                image=image.as_type[Self.display_dtype](),
                window_title=window_title,
            )
        else:
            Self._show(
                image=image.converted_as_type[ColorSpace.rgb, Self.display_dtype](),
                window_title=window_title,
            )

    @staticmethod
    fn _show[dtype: DType, color_space: ColorSpace, //](image: Image[color_space, dtype], var window_title: String):
        try:
            var show = _get_dylib_function[
                _libvisualizer,
                "show",
                fn (
                    data: UnsafePointer[UInt8, MutAnyOrigin],
                    height: c_int,
                    width: c_int,
                    channels: c_int,
                    window_title: UnsafePointer[c_char, ImmutAnyOrigin],
                ) -> None,
            ]()

            show(
                data=image.unsafe_uint8_ptr(),
                height=c_int(image.height()),
                width=c_int(image.width()),
                channels=c_int(image.channels()),
                window_title=window_title.as_c_string_slice().unsafe_ptr(),
            )
        except:
            abort()

    #
    # Run Loop
    #
    @staticmethod
    fn wait():
        _ = Self.wait(c_float.MAX_FINITE)

    @staticmethod
    fn wait(timeout: Float32) -> Bool:
        try:
            var wait = _get_dylib_function[_libvisualizer, "wait", fn (timeout: c_float) -> Bool]()

            return wait(c_float(timeout))
        except:
            abort()
