"""Email sending utilities for progress updates and notifications.

This module provides functionality for sending emails, particularly useful for monitoring
long-running computational jobs. It supports:
- Sending emails via Gmail SMTP
- Progress updates for iterative jobs
- Time estimation and completion tracking
- Secure password handling
- Configurable email sending (can be disabled)

Dependencies:
    - smtplib: For SMTP email sending
    - ssl: For secure SMTP connection
    - email.mime: For email message formatting
    - pathlib: For file path handling

Example Usage:
    ```python
    from email_sender import EmailSender
    
    # Initialize email sender
    sender = EmailSender(
        sender="your.email@gmail.com",
        reciever="recipient@example.com",
        pw="path/to/password/file.txt"
    )
    
    # Send progress update
    sender.email_update(
        start=time.time(),
        iter_start=time.time(),
        iter=5,
        total_iters=10,
        subject="Job Progress",
        params={"status": "running"}
    )
    ```
"""

import ssl
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from smtplib import SMTP_SSL
from typing import Callable, Dict, Optional

from utils import dict_to_str, timer


class EmailSender:
    """Email sender class for managing email communications.

    This class handles email configuration and sending, with support for progress
    updates and notifications. It can be configured to either send emails or
    operate in a no-send mode for testing.

    Attributes:
        sender: Email address of the sender
        reciever: Email address of the recipient
        pw: Password for the sender's email account
        send: Whether to actually send emails
        email: Function to send emails (either real or dummy)
    """

    def __init__(
        self,
        sender: Optional[str] = None,
        reciever: Optional[str] = None,
        pw: Optional[str] = None,
        send: bool = True,
        **kwargs,
    ) -> None:
        """Initialize EmailSender.

        Args:
            sender: Email address of the sender
            reciever: Email address of the recipient
            pw: Path to file containing email password
            send: Whether to actually send emails
            **kwargs: Additional keyword arguments (unused)
        """
        self.sender = sender
        self.reciever = reciever
        self.pw = self.retrieve_pw(pw)
        self.send = send
        if not self.send:
            print(f"Email settings: send set to {send}")

        # Set up email function based on configuration
        if None in (sender, reciever, pw):
            print(
                "At least one of email sender, reciever, or pw was not "
                "specified, will not send any emails."
            )
            self.email: Callable[[str, str], int] = lambda subject, content: 0
        else:
            self.email = self._email

    def retrieve_pw(self, file: Optional[str] = None) -> Optional[str]:
        """Retrieve the Gmail password from a file.

        Args:
            file: Path to the password file

        Returns:
            Password string if file exists, None otherwise
        """
        if file is None:
            return None
        with open(Path() / file, "r") as pw_file:
            pw = pw_file.read()
        return pw

    def _email(self, subject: str, content: str = "") -> None:
        """Send an email using the configured settings.

        Args:
            subject: Email subject line
            content: Email body content
        """
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
        params: Dict = {},
    ) -> None:
        """Send a progress update email for iterative computational jobs.

        This method sends an email with progress information including:
        - Number of iterations remaining
        - Completion percentage
        - Time taken for last iteration
        - Average time per iteration
        - Estimated time remaining
        - Additional parameters as JSON

        Note: Assumes iterations start at 0, so adds 1 to account for this.

        Args:
            start: Starting time of the entire job (from time.time())
            iter_start: Starting time of the last iteration (from time.time())
            iter: Index of the iteration that just finished
            total_iters: Total number of iterations
            subject: Email subject line
            params: Additional parameters to include in the email body

        Raises:
            AssertionError: If iter is negative
        """
        iter += 1
        assert iter > 0
        left = total_iters - iter
        done_percent = "{:.0f}".format((iter) / total_iters * 100)
        mean_time = (time.time() - start) / (iter)
        estimated_time_remaining = timer(left * mean_time)
        content = (
            f"{left} Experiments Left, {done_percent}% Completed\n"
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
    pw: Optional[str] = None,
    send: bool = True,
) -> None:
    """Send an email via Gmail SMTP.

    Args:
        content: Email body content
        subject: Email subject line
        sender: Sender's email address
        reciever: Recipient's email address
        pw: Gmail password for the sender's account
        send: Whether to actually send the email

    Raises:
        SMTPException: If there's an error connecting to the SMTP server
        Exception: For other email sending errors
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
