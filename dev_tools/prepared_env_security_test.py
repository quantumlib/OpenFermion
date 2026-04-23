import unittest
from unittest.mock import patch, MagicMock
from dev_tools.prepared_env import PreparedEnv
from dev_tools.github_repository import GithubRepository


class TestPreparedEnvSecurity(unittest.TestCase):
    @patch('requests.post')
    def test_report_status_to_github_token_in_header(self, mock_post):
        # Setup
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_post.return_value = mock_response

        repo = GithubRepository('my-org', 'my-repo', 'my-token')
        env = PreparedEnv(repo, 'my-commit', 'compare-commit', None, None)

        # Execute
        env.report_status_to_github('success', 'desc', 'ctx')

        # Verify
        args, kwargs = mock_post.call_args
        url = args[0]
        headers = kwargs.get('headers', {})

        # Security check: Token should NOT be in the URL
        self.assertNotIn('access_token=my-token', url, "Token should not be passed in the URL")

        # Security check: Token should be in the Authorization header
        self.assertEqual(
            headers.get('Authorization'),
            'Bearer my-token',
            "Token should be passed in the Authorization header",
        )


if __name__ == '__main__':
    unittest.main()
