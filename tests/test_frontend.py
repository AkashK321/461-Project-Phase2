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
        # Setup Chrome options for headless environment
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in background
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        # Initialize Driver
        self.driver = webdriver.Chrome(
            service=ChromeService(ChromeDriverManager().install()),
            options=chrome_options,
        )

        # Load the local HTML file
        file_path = os.path.abspath(
            "akashk321/461-project-phase2/461-Project-Phase2-frontend/frontend/index.html"
        )
        self.driver.get(f"file://{file_path}")

        yield

        # Teardown
        self.driver.quit()

    def test_login_ui_elements_exist(self):
        """Verify login inputs and buttons are present on load."""
        driver = self.driver
        # Check username and password fields exist
        assert driver.find_element(By.ID, "username").is_displayed()
        assert driver.find_element(By.ID, "password").is_displayed()

        # Verify dashboard is hidden initially
        dashboard = driver.find_element(By.ID, "dashboard")
        assert "hidden" in dashboard.get_attribute("class")

    def test_invalid_login_flow(self):
        """Test that invalid credentials show an error message."""
        driver = self.driver
        wait = WebDriverWait(driver, 5)

        # Input fake credentials
        driver.find_element(By.ID, "username").send_keys("fakeuser")
        driver.find_element(By.ID, "password").send_keys("badpass")

        # Click Login button (Finding by text since it has no ID in your HTML)
        login_btn = driver.find_element(By.XPATH, "//button[contains(text(), 'Login')]")
        login_btn.click()

        # Wait for error message to appear
        error_msg = wait.until(EC.visibility_of_element_located((By.ID, "login-error")))
        assert error_msg.is_displayed()

    def test_navigation_bar_state(self):
        """Verify nav buttons (Logout/Admin) are hidden initially."""
        driver = self.driver

        logout_btn = driver.find_element(By.ID, "logout-btn")
        admin_btn = driver.find_element(By.ID, "admin-btn")

        # Check for 'hidden' class defined in your CSS
        assert "hidden" in logout_btn.get_attribute("class")
        assert "hidden" in admin_btn.get_attribute("class")

    def test_upload_ui_interaction(self):
        """
        Simulate reaching the dashboard and interacting with Upload.
        """
        driver = self.driver

        # Javascript hack to show dashboard for testing UI elements inside it
        driver.execute_script(
            "document.getElementById('dashboard').classList.remove('hidden');"
        )

        # Verify Upload Section
        upload_input = driver.find_element(By.ID, "package-url")
        assert upload_input.is_displayed()

        upload_input.send_keys("https://github.com/fake/repo")

        # Find Upload button
        upload_btn = driver.find_element(
            By.XPATH, "//button[contains(text(), 'Upload')]"
        )
        assert upload_btn.is_displayed()

    def test_search_ui_interaction(self):
        """Simulate reaching the dashboard and interacting with Search."""
        driver = self.driver
        driver.execute_script(
            "document.getElementById('dashboard').classList.remove('hidden');"
        )

        search_input = driver.find_element(By.ID, "search-query")
        # Check default value is "**"
        assert search_input.get_attribute("value") == "**"

        search_input.clear()
        search_input.send_keys("React")

        search_btn = driver.find_element(
            By.XPATH, "//button[contains(text(), 'Search')]"
        )
        search_btn.click()
