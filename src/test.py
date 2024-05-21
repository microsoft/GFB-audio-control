import torch
from pathlib import Path
import hydra
import soundfile as sf
import utils.utils as utils
from utils.utils import apply_RIR_delay, crop_or_extend
from collections.abc import Sequence
import scipy.signal


def test(args):

    save_path = Path(args.save_path)
    # assert that this path exists
    assert save_path.exists(), "Save path does not exist!"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #######################
    # Load PyTorch Models #
    #######################
    model = hydra.utils.instantiate(args.backbone.dnn)
    model.to(device)

    # save checkpoints
    checkpoint_file=utils.get_checkpoint_file( args.checkpoint, save_path)
    checkpoint_file = Path(checkpoint_file)
    
    print("Checkpoint file:", checkpoint_file)

    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    steps = checkpoint["steps"]

    print(
        "Loaded checkpoint from {}, trained for {} steps".format(
                checkpoint_file.as_posix(), steps
        )
    )


    # enable cudnn autotuner to speed up training
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = args.exp.cudnn_autotuner
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.deterministic = False


    model.eval()

    diff = hydra.utils.instantiate(args.diff)

    do_test(
        args, diff=diff, model=model, device=device, save_path=save_path, iteration=0
    )


def do_test(args, diff=None, model=None, device=None, save_path=None, iteration=0):
    assert diff is not None, "diff must be provided"
    assert model is not None, "model must be provided"
    assert device is not None, "device must be provided"
    assert save_path is not None, "save_path must be provided"
    for m in args.tester.modes:
        if m == "unconditional":
            shape = (args.tester.batch, 1, args.tester.segment_length)
            result = diff.sample_unconditional(shape, model, args.tester.T, device)
            print("result", result.shape, result.mean(), result.std())
            result *= args.data.dataset.sigma_data

            utils.save_audio(
                    result, save_path, "unconditional", args.exp.fs
            )

        elif m == "bridge_reverb":
            print("testing bridge reverb")
            test_audio_example = args.tester.test_audio_example
            assert test_audio_example is not None, "test_audio_example must be provided"
            if isinstance(test_audio_example, Sequence):
                for i, example_audio in enumerate(args.tester.test_audio_example):
                    for j, example_RIR in enumerate(args.tester.test_RIR_example):
                        # prepare starting test example
                        audio, fs = sf.read(example_audio)
                        assert (
                            fs == args.exp.fs
                        ), "Test audio example must have the same sampling rate as the experiment"
                        audio = torch.Tensor(audio).to(device).float()
                        audio = crop_or_extend(
                            audio,
                            args.tester.segment_length,
                            deterministic=True,
                            start=8000,
                        )
                        assert (
                            audio.shape[-1] == args.tester.segment_length
                        ), "audio.shape[-1]={} != args.tester.segment_length={}".format(
                            audio.shape[-1], args.tester.segment_length
                        )
                        audio = audio / args.data.dataset.sigma_data

                        # load test RIR
                        h, fs = sf.read(example_RIR)
                        assert (
                            fs == args.exp.fs
                        ), "Test RIR example must have the same sampling rate as the experiment"
                        h = torch.Tensor(h).to(device).float()
                        # energy normalize the RIR
                        h = h / torch.sqrt(torch.sum(h**2))
                        # if h is even remove the last sample
                        if h.shape[0] % 2 == 0:
                            h = h[:-1]

                        x_in = apply_RIR_delay(audio.unsqueeze(0), h, 0).unsqueeze(0)
                        # attenuate a 3dB
                        x_in = x_in * 10 ** (-3 / 20)

                        # log that
                        utils.save_audio(
                                x_in * args.data.dataset.sigma_data,
                                save_path,
                                "x_in_bridge_T60_C50" + str(i) + str(j)+"_iter_"+str(iteration),
                                args.exp.fs
                        )

                        # params for conditional reverse sampling
                        T60s = torch.Tensor(args.tester.T60s)
                        C50s = torch.Tensor(args.tester.C50s)

                        batch = T60s.shape[0]
                        assert (
                            C50s.shape[0] == batch
                        ), "T60s and C50s must have the same length"
                        if args.tester.batch != batch:
                            print(
                                "Warning: batch size is different from the length of T60s and C50s. Using the length of T60s and C50s as batch size"
                            )

                        params = torch.cat(
                            [T60s.unsqueeze(1), C50s.unsqueeze(1)], dim=1
                        ).to(device)
                        result, z = diff.bridge(
                            x_in,
                            model=model,
                            Tsteps=args.tester.T,
                            cond=params,
                            cfg=args.tester.CFG,
                            schedule_type=args.tester.schedule,
                            bridge_end_t=args.tester.bridge_end_t,
                        )
                        print("result", result.shape, result.mean(), result.std())
                        result *= args.data.dataset.sigma_data

                        utils.save_audio(
                                result,
                                save_path,
                                "result_bridge_T60_C50" + str(i) + str(j)+"_iter_"+str(iteration),
                                args.exp.fs,
                        )


            elif isinstance(test_audio_example, str):
                raise NotImplementedError("test_audio_example must be a list of files")
        elif m == "bridge_clipping":
            print("test_audio_example", args.tester.test_audio_example)

            test_audio_example = args.tester.test_audio_example
            assert test_audio_example is not None, "test_audio_example must be provided"

            if isinstance(test_audio_example, Sequence):
                for i, example_audio in enumerate(args.tester.test_audio_example):
                    for j in range(len(args.tester.gain_db)):
                        # prepare starting test example
                        audio, fs = sf.read(example_audio)
                        if len(audio.shape) > 1:
                            audio = (audio[:, 0] + audio[:, 1]) / 2
                        if fs != args.exp.fs:
                            # resample
                            audio = scipy.signal.resample(
                                audio, int(len(audio) * args.exp.fs / fs)
                            )

                        audio = torch.Tensor(audio).to(device).float()
                        audio = crop_or_extend(
                            audio,
                            args.tester.segment_length,
                            deterministic=True,
                            start=8000,
                        )
                        assert (
                            audio.shape[-1] == args.tester.segment_length
                        ), "audio.shape[-1]={} != args.tester.segment_length={}".format(
                            audio.shape[-1], args.tester.segment_length
                        )

                        gain_db = args.tester.gain_db[j]


                        gain_lin = 10 ** (gain_db / 20)

                        y = torch.clamp(audio * gain_lin, min=-1, max=1)
                        y /= gain_lin

                        y = y / y.std()

                        x_in = y.unsqueeze(0).unsqueeze(0)

                        # log that
                        utils.save_audio(
                            x_in * args.data.dataset.sigma_data,
                            save_path,
                            "x_in_bridge_drive" + str(i) + str(j)+"_iter_"+str(iteration),
                            args.exp.fs,
                        )

                        # params for conditional reverse sampling
                        SDR = torch.Tensor(args.tester.SDR)

                        batch = SDR.shape[0]
                        if args.tester.batch != batch:
                            print(
                                "Warning: batch size is different from the length of T60s and C50s. Using the length of T60s and C50s as batch size"
                            )

                        # shape=(batch, 1, args.tester.segment_length)

                        params = torch.cat([SDR.unsqueeze(1)], dim=1).to(device)
                        result, z = diff.bridge(
                            x_in,
                            model=model,
                            Tsteps=args.tester.T,
                            cond=params,
                            cfg=args.tester.CFG,
                            schedule_type=args.tester.schedule,
                            bridge_end_t=args.tester.bridge_end_t,
                        )
                        print("result", result.shape, result.mean(), result.std())
                        result *= args.data.dataset.sigma_data

                        utils.save_audio(
                                result,
                                save_path,
                                "result_bridge_drive" + str(i) + str(j)+"_iter_"+str(iteration),
                                args.exp.fs,
                        )


            elif isinstance(test_audio_example, str):
                raise NotImplementedError("test_audio_example must be a list of files")
        else:
            raise NotImplementedError("Mode {} not implemented".format(m))


@hydra.main(config_path="conf", config_name="conf_speech_reverb", version_base="1.3.2")
def main(args):
    test(args)


if __name__ == "__main__":
    main()
