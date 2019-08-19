# For some cases, we need to compute steps per epoch (e.g. Keras's fit_generator asks for it).
# Hence, we should use respective computer for the underlying chunker. Therefore, we put these functions together.
# Although this basic chunker will be satisfactory for many cases, be sure that a corresponding steps_per_epoch_computer
# is utilized in case a custom chunker is used.


def create_chunks(sample_size, num_of_chunks, use_all=True):
    chunks = {}
    start = 0
    chunk_size = sample_size // num_of_chunks
    for i in range(num_of_chunks):
        chunks[i] = {"start": start, "end": start+chunk_size}
        start += chunk_size

    if use_all:
        chunks[num_of_chunks-1]["end"] = sample_size
        samples_per_epoch = sample_size
    else:
        samples_per_epoch = chunk_size * num_of_chunks

    return chunks, samples_per_epoch


def compute_steps_per_epoch(chunks, batch_size):
    steps_per_epoch = 0
    for elem in chunks:
        num_of_elements_in_chunk = chunks[elem]["end"] - chunks[elem]["start"]
        steps_per_epoch += num_of_elements_in_chunk // batch_size
        if num_of_elements_in_chunk % batch_size > 0:
            steps_per_epoch += 1

    return steps_per_epoch
