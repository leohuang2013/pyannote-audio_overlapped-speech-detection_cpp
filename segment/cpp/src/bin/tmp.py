def slide(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    window_size: int = round(self.duration * sample_rate) # 80000
    step_size: int = round(self.step * sample_rate) # 8000
    num_channels, num_samples = waveform.shape
    num_frames_per_chunk = 293 # Need check multiple wave file

    # prepare complete chunks
    if num_samples >= window_size:
        chunks: torch.Tensor = rearrange(
            waveform.unfold(1, window_size, step_size),
            "channel chunk frame -> chunk channel frame",
        )
        num_chunks, _, _ = chunks.shape
    else:
        num_chunks = 0 

    # prepare last incomplete chunk
    has_last_chunk = (num_samples < window_size) or (
        num_samples - window_size
    ) % step_size > 0 
    if has_last_chunk:
        last_chunk: torch.Tensor = waveform[:, num_chunks * step_size :]

    outputs: Union[List[np.ndarray], np.ndarray] = list()

    # slide over audio chunks in batch
    for c in np.arange(0, num_chunks, self.batch_size):
        batch: torch.Tensor = chunks[c : c + self.batch_size]
        outputs.append(self.infer(batch))

    # process orphan last chunk
    if has_last_chunk:
        # last_chunk 1x79674
        # last_output: numpy.array: 1x292x3
        last_output = self.infer(last_chunk[None])
        pad = num_frames_per_chunk - last_output.shape[1]
        last_output = np.pad(last_output, ((0, 0), (0, pad), (0, 0)))
        # after pad, last_output: numpy.array: 1x293x3

        outputs.append(last_output)
    # outputs in: list of numpy.array and first element is: 32x293x3
    # outputs out: list of numpy.array: 936x293x3
    outputs = np.vstack(outputs)

    return outputs

