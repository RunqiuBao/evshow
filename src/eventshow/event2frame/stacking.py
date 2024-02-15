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


"""
Copied from se-cff repo.
"""
class MixedDensityEventStacking:
    NO_VALUE = 0.
    STACK_LIST = ['stacked_polarity', 'index']

    def __init__(self, stack_size, num_of_event, height, width):
        self.stack_size = stack_size
        self.num_of_event = num_of_event
        self.height = height
        self.width = width

    def pre_stack(self, event_sequence, last_timestamp):
        x = event_sequence['x'].astype(numpy.int32)
        y = event_sequence['y'].astype(numpy.int32)
        p = 2 * event_sequence['p'].astype(numpy.int8) - 1
        t = event_sequence['t'].astype(numpy.int64)

        assert len(x) == len(y) == len(p) == len(t)

        past_mask = t <= last_timestamp
        p_x, p_y, p_p, p_t = x[past_mask], y[past_mask], p[past_mask], t[past_mask]
        p_t = p_t - p_t.min()
        past_stacked_event = self.make_stack(p_x, p_y, p_p, p_t)

        future_mask = t > last_timestamp
        if numpy.sum(future_mask) == 0:
            stacked_event_list = [past_stacked_event]
        else:
            f_x = x[future_mask][::-1]
            f_y = y[future_mask][::-1]
            f_p = p[future_mask][::-1]
            f_t = t[future_mask][::-1]
            f_p = f_p * -1
            f_t = f_t - f_t.min()
            f_t = f_t.max() - f_t
            future_stacked_event = self.make_stack(f_x, f_y, f_p, f_t)

            stacked_event_list = [past_stacked_event, future_stacked_event]

        return stacked_event_list

    def post_stack(self, pre_stacked_event):
        stacked_event_list = []
        for pf_stacked_event in pre_stacked_event:
            stacked_polarity = numpy.zeros([self.height, self.width, 1], dtype=numpy.float32)
            cur_stacked_event_list = []
            for stack_idx in range(self.stack_size - 1, -1, -1):
                stacked_polarity.put(pf_stacked_event['index'][stack_idx],
                                     pf_stacked_event['stacked_polarity'][stack_idx])
                cur_stacked_event_list.append(numpy.stack([stacked_polarity], axis=2))
            stacked_event_list.append(numpy.concatenate(cur_stacked_event_list[::-1], axis=2))
        if len(stacked_event_list) == 2:
            stacked_event_list[1] = stacked_event_list[1][:, :, ::-1, :]
        stacked_event = numpy.stack(stacked_event_list, axis=2)

        return stacked_event

    def make_stack(self, x, y, p, t):
        t = t - t.min()
        time_interval = t.max() - t.min() + 1
        t_s = (t / time_interval * 2) - 1.0
        stacked_event_list = {stack_value: [] for stack_value in self.STACK_LIST}
        cur_num_of_events = len(t)
        
        for _ in range(self.stack_size):
            stacked_event = self.stack_data(x, y, p, t_s)
            stacked_event_list['stacked_polarity'].append(stacked_event['stacked_polarity'])

            cur_num_of_events = cur_num_of_events // 2
            x = x[cur_num_of_events:]
            y = y[cur_num_of_events:]
            p = p[cur_num_of_events:]
            t_s = t_s[cur_num_of_events:]
            t = t[cur_num_of_events:]

        grid_x, grid_y = numpy.meshgrid(numpy.linspace(0, self.width - 1, self.width, dtype=numpy.int32),
                                     numpy.linspace(0, self.height - 1, self.height, dtype=numpy.int32))
        for stack_idx in range(self.stack_size - 1):
            prev_stack_polarity = stacked_event_list['stacked_polarity'][stack_idx]
            next_stack_polarity = stacked_event_list['stacked_polarity'][stack_idx + 1]

            assert numpy.all(next_stack_polarity[(prev_stack_polarity - next_stack_polarity) != 0] == 0)

            diff_stack_polarity = prev_stack_polarity - next_stack_polarity

            mask = diff_stack_polarity != 0
            stacked_event_list['index'].append((grid_y[mask] * self.width) + grid_x[mask])
            stacked_event_list['stacked_polarity'][stack_idx] = diff_stack_polarity[mask]

        last_stack_polarity = stacked_event_list['stacked_polarity'][self.stack_size - 1]
        mask = last_stack_polarity != 0
        stacked_event_list['index'].append((grid_y[mask] * self.width) + grid_x[mask])
        stacked_event_list['stacked_polarity'][self.stack_size - 1] = last_stack_polarity[mask]

        return stacked_event_list

    def stack_data(self, x, y, p, t_s):
        assert len(x) == len(y) == len(p) == len(t_s)

        stacked_polarity = numpy.zeros([self.height, self.width], dtype=numpy.int8)

        index = (y * self.width) + x
        
        stacked_polarity.put(index, p)

        stacked_event = {
            'stacked_polarity': stacked_polarity,
        }
        
        return stacked_event
 