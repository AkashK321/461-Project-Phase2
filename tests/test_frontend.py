import os
import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options


class TestFrontend:
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        self.driver = webdriver.Chrome(
            service=ChromeService(ChromeDriverManager().install()),
            options=chrome_options,
        )

        base_dir = os.path.dirname(os.path.abspath(__file__))
        html_path = os.path.join(base_dir, "..", "frontend", "index.html")
        html_path = os.path.abspath(html_path)

        if not os.path.exists(html_path):
            pytest.fail(f"HTML file not found at expected path: {html_path}")

        self.driver.get(f"file://{html_path}")

        yield

        self.driver.quit()

    def test_login_ui_elements_exist(self):
        driver = self.driver
        assert driver.find_element(By.ID, "username").is_displayed()
        assert driver.find_element(By.ID, "password").is_displayed()
        dashboard = driver.find_element(By.ID, "dashboard")
        assert "hidden" in dashboard.get_attribute("class")

    def test_invalid_login_flow(self):
        driver = self.driver
        wait = WebDriverWait(driver, 5)
        driver.find_element(By.ID, "username").send_keys("fakeuser")
        driver.find_element(By.ID, "password").send_keys("badpass")

        # Match button by text "Login"
        login_btn = driver.find_element(By.XPATH, "//button[normalize-space()='Login']")
        login_btn.click()

        error_msg = wait.until(EC.visibility_of_element_located((By.ID, "login-error")))
        assert error_msg.is_displayed()

    def test_navigation_bar_state(self):
        """Verify nav buttons (Logout/Admin) are hidden initially."""
        driver = self.driver

        # They should be inside the nav-bar which is hidden
        nav_bar = driver.find_element(By.ID, "nav-bar")
        assert "hidden" in nav_bar.get_attribute("class")

    def test_upload_ui_interaction(self):
        driver = self.driver
        driver.execute_script(
            "document.getElementById('dashboard').classList.remove('hidden');"
        )

        upload_input = driver.find_element(By.ID, "pkg-url")
        assert upload_input.is_displayed()

        upload_input.send_keys("https://github.com/fake/repo")
        upload_btn = driver.find_element(
            By.XPATH, "//button[contains(text(), 'Upload')]"
        )
        assert upload_btn.is_displayed()

    def test_search_ui_interaction(self):
        driver = self.driver
        driver.execute_script(
            "document.getElementById('dashboard').classList.remove('hidden');"
        )

        search_input = driver.find_element(By.ID, "search-q")

        # Check default value (regex string)
        assert search_input.get_attribute("value") == ".*"

        search_input.clear()
        search_input.send_keys("React")

        search_btn = driver.find_element(
            By.XPATH, "//button[contains(text(), 'Search')]"
        )
        search_btn.click()
