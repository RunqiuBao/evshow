import numpy


def AccumulateEventsIntoFrame(events, frameShape):
    """
    simply accumulate events into frame based representation.
    """
    eventFrame = numpy.zeros(frameShape[::-1], dtype='int8')
    xs, ys, ps, ts = events['x'].astype('int'), events['y'].astype('int'), events['p'].astype('int8'), events['t']
    xsMask = numpy.where(xs < frameShape[0], True, False)
    ysMask = numpy.where(ys < frameShape[1], True, False)
    mask = numpy.logical_and(xsMask, ysMask)
    xs, ys, ps, ts = xs[mask], ys[mask], ps[mask], ts[mask]
    ps[numpy.where(ps == 0)] = -1
    try:
        numpy.add.at(eventFrame, (ys, xs), ps)
    except Exception as e:
        print('Error: events batch is empty.\n', e)
        return None, None

    eventFrameImg = eventFrame.astype('int16')
    eventFrameImg -= eventFrameImg.min()
    return (eventFrameImg.astype('float') * 255 / eventFrameImg.max()).astype('uint8'), eventFrame