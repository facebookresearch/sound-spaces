import logging
import requests


URLS = {
    "update_submission_data": "/api/jobs/challenge/{}/update_submission/",
}


class EvalAI_Interface:
    def __init__(
            self,
            AUTH_TOKEN,
            EVALAI_API_SERVER
            ):
        self.AUTH_TOKEN = AUTH_TOKEN
        self.EVALAI_API_SERVER = EVALAI_API_SERVER

    def get_request_headers(self):
        headers = {"Authorization": "Bearer {}".format(self.AUTH_TOKEN)}
        return headers

    def make_request(self, url, method, data=None):
        headers = self.get_request_headers()
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                data=data,
                timeout=200
            )
            response.raise_for_status()
            print("Successful Status", response.json())
        except requests.exceptions.RequestException as e:
            import traceback
            print(traceback.print_exc())
            print("The worker is not able to establish connection with EvalAI", response.json())
            raise
        return response.json()

    def return_url_per_environment(self, url):
        base_url = "{0}".format(self.EVALAI_API_SERVER)
        url = "{0}{1}".format(base_url, url)
        return url

    def update_submission_data(self, data, challenge_pk):
        url = URLS.get("update_submission_data").format(challenge_pk)
        url = self.return_url_per_environment(url)
        response = self.make_request(url, "PUT", data=data)
        return response