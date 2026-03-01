#
# size.mojo
# mosaic
#
# Created by Christian Bator on 03/25/2025
#


@fieldwise_init
struct Size(Copyable, Movable):
    var height: Int
    var width: Int
