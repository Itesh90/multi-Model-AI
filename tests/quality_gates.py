import pytest
import coverage
import subprocess
import json
import os
from datetime import datetime

def test_code_coverage():
    """Ensure code coverage meets minimum requirements"""
    # Run coverage report
    cov = coverage.Coverage()
    cov.load()
    total_cov = cov.report()
    
    print(f"\nTotal code coverage: {total_cov:.2f}%")
    
    # Check coverage thresholds
    assert total_cov >= 80, f"Code coverage {total_cov:.2f}% below minimum 80%"
    
    # Get detailed coverage data
    data = cov.get_data()
    analysis = {}
    
    for filename in data.measured_files():
        _, statements, excluded, missing, _ = data.analysis(filename)
        percent = (len(statements) - len(missing)) / len(statements) * 100 if statements else 100
        analysis[filename] = {
            "coverage": percent,
            "missing_lines": missing
        }
        
        # Check individual file coverage
        if percent < 70:
            print(f"Warning: {filename} has low coverage ({percent:.2f}%)")
    
    # Save detailed report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"coverage_detailed_{timestamp}.json"
    with open(report_file, "w") as f:
        json.dump(analysis, f, indent=2)
    
    print(f"Detailed coverage report saved to {report_file}")

def test_security_vulnerabilities():
    """Check for known security vulnerabilities"""
    try:
        # Use Safety to check for vulnerabilities
        result = subprocess.run(
            ["safety", "check", "--json"],
            capture_output=True,
            text=True
        )
        
        vulnerabilities = json.loads(result.stdout)
        
        # Check if there are vulnerabilities
        if vulnerabilities["vulnerabilities"]:
            print("\nSecurity vulnerabilities found:")
            for vuln in vulnerabilities["vulnerabilities"]:
                print(f"- {vuln['package']} {vuln['version']}: {vuln['advisory']}")
            
            assert False, f"{len(vulnerabilities['vulnerabilities'])} security vulnerabilities found"
        else:
            print("\nNo security vulnerabilities found")
    
    except FileNotFoundError:
        pytest.skip("Safety not installed. Run 'pip install safety' to enable security checks")

def test_performance_metrics():
    """Check performance metrics against baselines"""
    # In a real implementation, you'd load previous performance metrics
    # For students, we'll use hardcoded baselines
    
    # Baseline response times (seconds)
    BASELINES = {
        "/text/sentiment": 0.5,
        "/text/embedding": 0.8,
        "/process-image": 2.0
    }
    
    # Current performance (in a real test, these would come from performance tests)
    CURRENT = {
        "/text/sentiment": 0.45,
        "/text/embedding": 0.75,
        "/process-image": 1.9
    }
    
    print("\nPerformance Metrics Comparison:")
    for endpoint, current in CURRENT.items():
        baseline = BASELINES[endpoint]
        change = ((current - baseline) / baseline) * 100
        
        status = "✅" if change <= 10 else "⚠️" if change <= 20 else "❌"
        print(f"{status} {endpoint}: {current:.2f}s (baseline: {baseline:.2f}s, change: {change:.1f}%)")
        
        # Allow up to 10% regression
        assert change <= 10, f"{endpoint} performance regressed by {change:.1f}%"

# To run these quality gates as part of your CI/CD:
# pytest tests/quality_gates.py