

module.exports = function calculateAge(birthNumber) {
    const yearPrefix = birthNumber.substring(0, 2) >= '22' ? '19' : '20'; // 2000년 이전과 이후 구분
    const fullYear = yearPrefix + birthNumber.substring(0, 2);
    const month = birthNumber.substring(2, 4);
    const day = birthNumber.substring(4, 6);
  
    const today = new Date();
    const birthDate = new Date(`${fullYear}-${month}-${day}`);
  
    let age = today.getFullYear() - birthDate.getFullYear();
    const m = today.getMonth() - birthDate.getMonth();
    if (m < 0 || (m === 0 && today.getDate() < birthDate.getDate())) {
      age--;
    }
  
    return age;
  }
  