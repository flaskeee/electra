import fire

import build_pretraining_dataset
import glob


def main(
        input_files: str,
        vocab_file: str,
        out_dir: str,
        max_seq_length=128,
        do_lower_case=True,
):
    example_writer = build_pretraining_dataset.ExampleWriter(
        job_id=0,
        vocab_file=vocab_file,
        output_dir=out_dir,
        max_seq_length=max_seq_length,
        num_jobs=1,
        blanks_separate_docs=False,
        do_lower_case=do_lower_case,
        strip_accents=True,
    )

    for f in glob.glob(input_files):
        example_writer.write_examples(open(f), is_lines=True)
    print('done!')


if __name__ == '__main__':
    fire.Fire(main)
