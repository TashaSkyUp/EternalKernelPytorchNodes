import copy

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def ETK_descript_base(cls):
    cls.FUNCTION = "func"
    cls.CATEGORY = "ETK/Descript"
    NODE_CLASS_MAPPINGS[cls.__name__] = cls
    return cls


@ETK_descript_base
class DecodeTranscriptNode:
    """
    Decode one of descript's transcript formatted files
    String input is the files contents
    output is 3 lists of strings (speaker,duration,paragraph,)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": ("STRING", {"multiline": True,
                                      "default": ""
                                      }),
            }
        }

    RETURN_TYPES = ("LIST", "LIST", "LIST")
    RETURN_NAMES = ("speakers", "durations", "paragraphs")
    OUTPUT_NODE = True

    def func(self, string):
        """
        input is like:
        [00:00:00] jackie: Let's talk about a

        [00:00:01] jackie: thing

        :param string:
        :return:
        """
        paras = string.split("\n\n")
        speakers = []
        durations = []
        paragraphs = []

        for para in paras:
            if para.strip() == "":
                continue
            # first find the first ] and split on that
            part1 = para.find("]")
            time_part = para[:part1 + 1]
            remainder = para[part1 + 1:]

            # now find first : in the remainder
            part2 = remainder.find(":")
            speaker_part = remainder[:part2 + 1].replace(":", "").strip()
            paragraph_part = remainder[part2 + 1:]
            speakers.append(speaker_part.strip())

            # now just remove [ and ] from time_part
            time_part = time_part.replace("[", "").replace("]", "").strip()

            # split it on :
            time_part = time_part.split(":")

            # now convert to seconds
            time_part = int(time_part[0]) * 60 * 60 + int(time_part[1]) * 60 + int(time_part[2])

            # save here for post-processing later
            durations.append(time_part)

            paragraphs.append(paragraph_part.strip())

        import copy
        start_seconds = copy.copy(durations)
        durations = []
        for i in range(len(start_seconds) - 1):
            durations.append(start_seconds[i + 1] - start_seconds[i])
        # now append the last value
        durations.append(int(sum(durations) / len(durations)))

        return (speakers, durations, paragraphs,)
