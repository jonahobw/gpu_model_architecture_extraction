import json
import ssl
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from smtplib import SMTP_SSL, SMTPException

from utils import dict_to_str, timer


class EmailSender:
    """Stores email sender, reciever, and pw."""

    def __init__(
        self,
        sender: str = None,
        reciever: str = None,
        pw: str = None,
        send: bool = True,
        **kwargs,
    ) -> None:
        """Store email params."""

        self.sender = sender
        self.reciever = reciever
        self.pw = self.retrieve_pw(pw)
        self.send = send
        if not self.send:
            print(f"Email settings: send set to {send}")
        # dummy function in case an argument is not provided:
        if None in (sender, reciever, pw):
            print(
                "At least one of email sender, reciever, or pw was not"
                "specified, will not send any emails."
            )
            self.email = lambda subject, content: 0
        else:
            self.email = self._email

    def retrieve_pw(self, file: str = None) -> str:
        """Retrieves the gmail password from a file."""

        if file is None:
            return None
        with open(Path() / file, "r") as pw_file:
            pw = pw_file.read()
        return pw

    def _email(self, subject, content=""):
        """Send an email."""
        email(
            content=content,
            subject=subject,
            sender=self.sender,
            reciever=self.reciever,
            pw=self.pw,
            send=self.send,
        )

    def email_update(
        self,
        start: float,
        iter_start: float,
        iter: int,
        total_iters: int,
        subject: str,
        params: dict = {},
    ) -> None:
        """
        Format and send an email that gives a progress update on iterative computational jobs.
        One email will be sent per iteration, and in each email, the message will include
        - the time for the last iteration
        - the average time per iteration
        - an estimate of how much longer until the job is done

        **Assumes that the iterations start at 0, not 1, so adds 1 to account for this

        Args:
            start (float): the starting time of the entire job, obtained from time.time()
            iter_start (float): the starting time of the iteration that just ended, obtained
                from time.time()
            iter (int): the iteration that just finished
            total_iters (int): how many iterations there are int total
            subject (str): the email subject line
            params (dict): any json data to be added to the email body
        """
        iter += 1
        assert iter > 0
        left = total_iters - iter
        done_percent = "{:.0f}".format((iter) / total_iters * 100)
        mean_time = (time.time() - start) / (iter)
        estimated_time_remaining = timer(left * mean_time)
        content = (
            f"{left} Experiments Left, {done_percent}% Completed"
            f"Time of last experiment: {timer(time.time() - iter_start)}\n"
            f"Estimated time remaining ({left} experiments left and "
            f"{timer(mean_time)} per experiment): "
            f"{estimated_time_remaining}\n\n"
            f"{dict_to_str(params)}\n"
        )
        self.email(subject, content)


def email(
    content: str,
    subject: str,
    sender: str,
    reciever: str,
    pw: str = None,
    send: bool = True,
) -> None:
    """
    Sends an email from a gmail account.
    :param content: the message inside the email.
    :param subject: the subject line.
    :param sender: the sending email address.
    :param reciever: the destination email address.
    :param pw: the gmail password for the sending email address.
    :param send: will only send an email if this is true.
    :return: None
    """

    if not send:
        return

    message = MIMEMultipart()
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = reciever
    message.attach(MIMEText(content, "plain"))

    try:
        context = ssl.create_default_context()
        with SMTP_SSL(host="smtp.gmail.com", port=465, context=context) as server:
            server.login(sender, pw)
            server.sendmail(sender, reciever, message.as_string())
            server.quit()
    except Exception as e:
        print("Error while trying to send email: \n%s", e)
