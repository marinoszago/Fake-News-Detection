import subprocess
import shlex

def RateSentiment(sentiString):

    sentiString = sentiString.replace(" ", "+")

    # open a subprocess using shlex to get the command line string into the correct args list format
    p = subprocess.Popen(shlex.split("java -jar E:/senti/SentiStrengthCom.jar stdin sentidata "

                                     "E:/senti/SentiStrength_Data/"), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    # communicate via stdin the string to be rated. Note that all spaces are replaced with +
    stdout_text, stderr_text = p.communicate(sentiString)
    # remove the tab spacing between the positive and negative ratings. e.g. 1    -5 -> 1-5
    stdout_text = stdout_text.rstrip().replace("\t", "")

    pos = []
    neg = []
    stdout_text = stdout_text.splitlines()
    for item in stdout_text:
        results = item.split("-")
        pos.append(int(results[0]))
        neg.append(int(results[1]))

    try:
        pos = float(sum(pos)) / float(len(stdout_text))
        neg = float(sum(neg)) / float(len(stdout_text))
    except ZeroDivisionError:
        pos = 0
        neg = 0

    return pos, neg
