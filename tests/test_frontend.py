import os
import pytest
import time
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
        # Ensure this points to where you saved the split files
        html_path = os.path.join(base_dir, "..", "frontend", "index.html")
        html_path = os.path.abspath(html_path)

        if not os.path.exists(html_path):
            pytest.fail(f"HTML file not found at expected path: {html_path}")

        self.driver.get(f"file://{html_path}")
        
        # Allow a brief moment for the app.js router to initialize
        time.sleep(0.5)

        yield

        self.driver.quit()

    def test_login_ui_elements_exist(self):
        driver = self.driver
        # Should default to #/login
        assert driver.find_element(By.ID, "username").is_displayed()
        assert driver.find_element(By.ID, "password").is_displayed()
        
        dashboard = driver.find_element(By.ID, "dashboard")
        assert "hidden" in dashboard.get_attribute("class")

    def test_routing_logic(self):
        """Test that changing the hash routes to the correct view."""
        driver = self.driver
        
        # Mock a logged-in state so the Router allows access to dashboard
        driver.execute_script("authToken = 'mock_token';")
        
        # Navigate using the Hash, which tests your Router logic
        driver.execute_script("window.location.hash = '#/dashboard';")
        time.sleep(0.5) # Wait for hashchange event
        
        dashboard = driver.find_element(By.ID, "dashboard")
        login_section = driver.find_element(By.ID, "login-section")
        
        # Dashboard should be visible, Login hidden
        assert "hidden" not in dashboard.get_attribute("class")
        assert "hidden" in login_section.get_attribute("class")

    def test_admin_modals_routing(self):
        """Test that admin routes open the corresponding modals."""
        driver = self.driver
        driver.execute_script("authToken = 'mock_token';")
        
        # 1. Test Create User Modal
        driver.execute_script("window.location.hash = '#/admin/create-user';")
        time.sleep(0.5)
        
        modal = driver.find_element(By.ID, "create-user-modal")
        # Check if the dialog is open (HTML5 dialog uses 'open' attribute)
        assert modal.get_attribute("open") is not None

        # 2. Test View Users Modal
        driver.execute_script("window.location.hash = '#/admin/view-users';")
        time.sleep(0.5)
        
        modal_view = driver.find_element(By.ID, "view-users-modal")
        assert modal_view.get_attribute("open") is not None

    def test_invalid_login_flow(self):
        driver = self.driver
        wait = WebDriverWait(driver, 5)
        
        # Ensure we are on login page
        driver.execute_script("window.location.hash = '#/login';")
        
        driver.find_element(By.ID, "username").send_keys("fakeuser")
        driver.find_element(By.ID, "password").send_keys("badpass")

        login_btn = driver.find_element(By.XPATH, "//button[normalize-space()='Login']")
        login_btn.click()

        # The error logic is in app.js, so this confirms app.js is loaded correctly
        error_msg = wait.until(EC.visibility_of_element_located((By.ID, "login-error")))
        assert error_msg.is_displayed()

    def test_search_ui_interaction(self):
        driver = self.driver
        
        # Bypass login via router hash + mock token
        driver.execute_script("authToken = 'mock_token';")
        driver.execute_script("window.location.hash = '#/dashboard';")
        time.sleep(0.5)

        search_input = driver.find_element(By.ID, "search-q")
        assert search_input.get_attribute("value") == ".*"

        search_input.clear()
        search_input.send_keys("React")

        search_btn = driver.find_element(By.XPATH, "//button[contains(text(), 'Search')]")
        search_btn.click()
        