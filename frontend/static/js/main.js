        // ==================== NAVEGACIÓN Y TRANSICIONES ====================
        document.addEventListener('DOMContentLoaded', function() {
            const pageTransition = document.getElementById('pageTransition');
            const navTabs = document.querySelectorAll('.nav-tab');
            
            // Marcar pestaña activa basada en la URL actual
            function setActiveTab() {
                const currentPath = window.location.pathname;
                navTabs.forEach(tab => {
                    const tabPath = new URL(tab.href).pathname;
                    if (tabPath === currentPath || (currentPath === '/' && tabPath === '/index')) {
                        tab.classList.add('active');
                    } else {
                        tab.classList.remove('active');
                    }
                });
            }

            // Función de transición entre páginas
            function navigateToPage(url) {
                pageTransition.classList.add('active');
                
                setTimeout(() => {
                    window.location.href = url;
                }, 300);
            }

            // Event listeners para navegación
            navTabs.forEach(tab => {
                tab.addEventListener('click', function(e) {
                    e.preventDefault();
                    const url = this.href;
                    
                    // Si ya estamos en la página, no hacer nada
                    if (this.classList.contains('active')) {
                        return;
                    }
                    
                    navigateToPage(url);
                });
            });

            // Establecer pestaña activa al cargar
            setActiveTab();

            // ==================== ANIMACIONES DE BOTONES ====================
            const buttons = document.querySelectorAll('.btn, .nav-tab, .footer-icon');
            
            buttons.forEach(button => {
                button.addEventListener('mouseenter', function() {
                    this.style.transform = 'translateY(-3px) scale(1.05)';
                });
                
                button.addEventListener('mouseleave', function() {
                    this.style.transform = '';
                });
            });

            // ==================== EFECTOS DE SCROLL ====================
            const cards = document.querySelectorAll('.card');
            
            const observerOptions = {
                threshold: 0.1,
                rootMargin: '0px 0px -50px 0px'
            };
            
            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        entry.target.style.animation = 'fadeInUp 0.6s ease-out forwards';
                        entry.target.style.opacity = '1';
                    }
                });
            }, observerOptions);
            
            cards.forEach(card => {
                card.style.opacity = '0';
                observer.observe(card);
            });

            // ==================== EFECTOS ADICIONALES ====================
            
            // Efecto de partículas en el fondo (sutil)
            function createParticle() {
                const particle = document.createElement('div');
                particle.style.cssText = `
                    position: fixed;
                    width: 4px;
                    height: 4px;
                    background: rgba(99, 102, 241, 0.3);
                    border-radius: 50%;
                    pointer-events: none;
                    z-index: 1;
                    left: ${Math.random() * 100}vw;
                    top: 100vh;
                    animation: float-up 8s linear forwards;
                `;
                
                document.body.appendChild(particle);
                
                setTimeout(() => {
                    particle.remove();
                }, 8000);
            }

            // Crear partículas ocasionalmente
            setInterval(createParticle, 3000);

            // CSS para animación de partículas
            const style = document.createElement('style');
            style.textContent = `
                @keyframes float-up {
                    to {
                        transform: translateY(-100vh) rotate(360deg);
                        opacity: 0;
                    }
                }
            `;
            document.head.appendChild(style);

            // ==================== TECLADO SHORTCUTS ====================
            document.addEventListener('keydown', function(e) {
                // Alt + número para navegación rápida
                if (e.altKey && e.key >= '1' && e.key <= '8') {
                    e.preventDefault();
                    const tabIndex = parseInt(e.key) - 1;
                    if (navTabs[tabIndex]) {
                        navigateToPage(navTabs[tabIndex].href);
                    }
                }
            });

            // ==================== INDICADOR DE CARGA ====================
            window.addEventListener('beforeunload', function() {
                pageTransition.classList.add('active');
            });

            // Ocultar transición cuando la página carga completamente
            window.addEventListener('load', function() {
                setTimeout(() => {
                    pageTransition.classList.remove('active');
                }, 300);
            });
        });

        // ==================== FUNCIONES GLOBALES ====================
        
        // Función para mostrar notificaciones
        function showNotification(message, type = 'info') {
            const notification = document.createElement('div');
            notification.className = `notification notification-${type}`;
            notification.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 1rem 1.5rem;
                background: var(--bg-card);
                border: 1px solid var(--border-color);
                border-radius: var(--radius-lg);
                color: var(--text-primary);
                box-shadow: var(--shadow-lg);
                z-index: 10000;
                transform: translateX(400px);
                transition: transform var(--transition-normal);
            `;
            
            if (type === 'success') {
                notification.style.borderColor = 'var(--success-color)';
            } else if (type === 'error') {
                notification.style.borderColor = 'var(--error-color)';
            } else if (type === 'warning') {
                notification.style.borderColor = 'var(--warning-color)';
            }
            
            notification.innerHTML = `
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <i class="fas fa-${type === 'success' ? 'check' : type === 'error' ? 'times' : type === 'warning' ? 'exclamation' : 'info'}-circle"></i>
                    <span>${message}</span>
                </div>
            `;
            
            document.body.appendChild(notification);
            
            setTimeout(() => {
                notification.style.transform = 'translateX(0)';
            }, 100);
            
            setTimeout(() => {
                notification.style.transform = 'translateX(400px)';
                setTimeout(() => notification.remove(), 300);
            }, 3000);
        }

        // Función para modo oscuro/claro (ya implementado por defecto en modo oscuro)
        function toggleTheme() {
            // Implementar si se desea soporte para tema claro
            console.log('Función de cambio de tema disponible para implementar');
        }