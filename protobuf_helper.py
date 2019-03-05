# Some function to get hyperparameter using protobuf, only for those in list format

# Changed 'google.protobuf.pyext._message.RepeatedScalarContainer' to list
# options is dict which the repeated_scalar contain index of
def protobuf_to_list(repeated_scalar, options=None):
    output = list()
    for elem in repeated_scalar:
        if options is None:
            output.append(elem)
        else:
            output.append(options[elem])
    return output


# Specifically set for transforming
def protobuf_to_channels(repeated_channel):
    channel_list = list()
    for channel in repeated_channel:
        ch = list()
        for layers in channel.channel:
            ch.append(layers)
        channel_list.append(ch)
    return channel_list
