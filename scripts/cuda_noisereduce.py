'''
from numba import cuda

@cuda.jit
def reduce_noise(
    y, 
    sr, 
    stationary=False,
    y_noise=None,
    prop_decrease=1.0,
    time_constant_s=2.0,
    freq_mask_smooth_hz=500,
    time_mask_smooth_ms=50,
    thresh_n_mult_nonstationary=2,
    sigmoid_slope_nonstationary=10,
    n_std_thresh_stationary = 1.5,
    tmp_folder=None,
    chunk_size=600000,
    padding=30000,
    n_fft=1024,
    win_length=None,
    hop_length=None,
    clip_noise_stationary = True,
    use_tqdm=False,
    n_jobs = 1
):
    n_channels, n_frames = y.shape
    ich1 = 
        
    

    def get_traces(self, start_frame=None, end_frame=None):
        """Grab filtered data iterating over chunks"""
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.n_frames

        if self._chunk_size is not None:
            if end_frame - start_frame > self._chunk_size:
                ich1 = int(start_frame / self._chunk_size)
                ich2 = int((end_frame - 1) / self._chunk_size)
                
                # write output to temp memmap for parallelization
                with tempfile.NamedTemporaryFile(prefix=self._tmp_folder) as fp:
                    # create temp file
                    filtered_chunk = np.memmap(
                        fp,
                        dtype=self._dtype,
                        shape=(self.n_channels, int(end_frame - start_frame)),
                        mode="w+",
                    )
                    pos_list = []
                    start_list = []
                    end_list = []
                    pos = 0
                    for ich in range(ich1, ich2 + 1):
                        if ich == ich1:
                            start0 = start_frame - ich * self._chunk_size
                        else:
                            start0 = 0
                        if ich == ich2:
                            end0 = end_frame - ich * self._chunk_size
                        else:
                            end0 = self._chunk_size
                        pos_list.append(pos)
                        start_list.append(start0)
                        end_list.append(end0)
                        pos += end0 - start0
                    
                    Parallel(n_jobs=self.n_jobs)(delayed(self._iterate_chunk)(filtered_chunk, pos, end0, start0, ich)
                       for pos, start0, end0, ich in zip(
                        tqdm(pos_list, disable=not(self.use_tqdm)), start_list, end_list, range(ich1, ich2 + 1)
                    )
                   )
                    if self.flat:
                        return filtered_chunk.astype(self._dtype).flatten()
                    else:
                        return filtered_chunk.astype(self._dtype)

        filtered_chunk = self.filter_chunk(start_frame=0, end_frame=end_frame)
        if self.flat:
            return filtered_chunk.astype(self._dtype).flatten()
        else:
            return filtered_chunk.astype(self._dtype)
    
'''